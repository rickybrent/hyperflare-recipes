# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Parallel draft proposals, stochastic target verification and accepted-state commits."""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from hyperflare_recipes.components.collectives import PreparedAllReduce
from ignition.preparation.persistent import persistent_copy
from .draft_sampling import DRAFT_CANDIDATES, PreparedDraftDistribution, PreparedDraftBlockBroadcast
from .paged_batch import Metadata, prepare_batch, row_rotary_offsets
from .speculation import (
    DRAFT_RNG_DOMAIN,
    PreparedSpeculativeCommit,
    PreparedSpeculativeDistributions,
    PreparedStochasticVerification,
)
from ignition.components.state import save_hidden


def prepare_sampling_broadcasts(proposals, verification, *, collectives, session, name, result_runtime=None):
    """Synchronize proposals before dependent work and acceptance before state commits."""
    operations = [(f"proposal{index}", proposal) for index, proposal in enumerate(proposals)]
    if verification is not None:
        operations.append(("acceptance", verification))
    runtime = collectives["vocabulary_runtime"] if result_runtime is None else result_runtime
    for suffix, operation in operations:
        operation.prepare_broadcast(runtime=runtime, session=session,
            coordinator=collectives["coordinator"], name=f"{name}.{suffix}")


@dataclass(frozen=True)
class DraftProposalState:
    before: torch.Tensor
    active: torch.Tensor
    candidates: torch.Tensor
    drafts: torch.Tensor
    valid: torch.Tensor
    base_offset: torch.Tensor
    metadata: Metadata | None
    features: dict[str, torch.Tensor] | None = None
    # Each row's rotary offset from its token index, for a pool that declares the region.
    rotary_offsets: torch.Tensor | None = None
    tile_masses: torch.Tensor | None = None
    copy_offered_mask: torch.Tensor | None = None
    confidence: torch.Tensor | None = None


def _resolve_copy_proposals(owner, active, safe, before, limits, depth):
    """Validate per-slot host copy proposals and return contiguous prefix masks and token IDs."""
    copy_ids = getattr(owner, "copy_ids", None)
    if copy_ids is None or depth <= 0:
        return None, None
    device = safe.device
    copy_count = owner.copy_count[safe]
    copy_expected = owner.copy_expected[safe]
    copy_generation = owner.copy_generation[safe]
    row_valid = (
        active
        & (copy_generation == owner.requests.generation[safe])
        & (copy_expected == before)
        & (copy_count > 0)
    )
    grammar_enabled = getattr(owner.requests, "grammar_enabled", None)
    if grammar_enabled is not None:
        row_valid = row_valid & (grammar_enabled[safe] == 0)
    copy_width = copy_ids.shape[1]
    take = min(depth, copy_width)
    tokens = copy_ids[safe, :take]
    if take < depth:
        tokens = torch.nn.functional.pad(tokens, (0, depth - take), value=-1)
    positions = torch.arange(depth, device=device)
    raw = (
        row_valid[:, None]
        & (positions[None, :] < copy_count[:, None])
        & (positions[None, :] < (limits[:, None] - 1))
        & (tokens >= 0)
        & (tokens < owner.requests.vocab_size)
    )
    # Copy proposals form a contiguous prefix starting at draft position zero.
    copy_active = raw.to(torch.int32).cumprod(dim=1).bool()
    return copy_active, tokens


def _copy_offered_mask(copy_active):
    """Pack contiguous copy-draft active flags into a per-row bitmask."""
    positions = torch.arange(copy_active.shape[1], device=copy_active.device, dtype=torch.int64)
    return (copy_active.to(torch.int64) << positions[None, :]).sum(dim=1)


@triton.jit
def _apply_copy_proposals_kernel(
    Drafts,
    Probabilities,
    Valid,
    TileMass,
    CopyActive,
    CopyTokens,
    VocabStartPtr,
    VOCAB_START_CONST: tl.constexpr,
    USE_VOCAB_START_PTR: tl.constexpr,
    LOCAL_VOCAB: tl.constexpr,
    BLOCKS: tl.constexpr,
    TILE_BLOCK: tl.constexpr,
    HAS_MASSES: tl.constexpr,
):
    tile = tl.program_id(0)
    row_depth = tl.program_id(1)
    if tl.load(CopyActive + row_depth) == 0:
        return
    copy_tok = tl.load(CopyTokens + row_depth)
    if tile == 0:
        tl.store(Drafts + row_depth, copy_tok)
        tl.store(Valid + row_depth, True)
    vocab_start = (
        tl.load(VocabStartPtr).to(tl.int64)
        if USE_VOCAB_START_PTR
        else tl.full((), VOCAB_START_CONST, tl.int64)
    )
    local_target = copy_tok - vocab_start
    col = tile * TILE_BLOCK + tl.arange(0, TILE_BLOCK)
    mask = col < LOCAL_VOCAB
    point_mass = (col.to(tl.int64) == local_target).to(Probabilities.dtype.element_ty)
    tl.store(Probabilities + row_depth * LOCAL_VOCAB + col, point_mass, mask=mask)
    if HAS_MASSES:
        tile_start = tile * TILE_BLOCK
        tile_has_target = (
            (local_target >= tile_start)
            & (local_target < tile_start + TILE_BLOCK)
            & (local_target < LOCAL_VOCAB)
        )
        tl.store(
            TileMass + row_depth * BLOCKS + tile,
            tl.where(tile_has_target, 1.0, 0.0).to(TileMass.dtype.element_ty),
        )


def _apply_copy_proposals(
    drafts,
    probabilities,
    valid,
    copy_active,
    copy_tokens,
    *,
    vocab_start=0,
    tile_masses=None,
    tile_block=256,
):
    """Overlay contiguous copy-draft prefixes with point-mass q=1.0 proposal distributions."""
    batch, depth = drafts.shape
    local_vocab = probabilities.shape[-1]
    device = drafts.device
    blocks = (local_vocab + tile_block - 1) // tile_block if tile_block > 0 else 0
    if (
        probabilities.is_cuda
        and batch > 0
        and depth > 0
        and local_vocab > 0
        and probabilities.is_contiguous()
        and drafts.is_contiguous()
        and valid.is_contiguous()
        and tile_block > 0
        and (tile_block & (tile_block - 1)) == 0
        and (tile_masses is None or (tile_masses.is_contiguous() and tile_masses.shape[-1] == blocks))
    ):
        from torch.fx.experimental.proxy_tensor import get_proxy_mode

        use_ptr = isinstance(vocab_start, torch.Tensor)
        vocab_start_ptr = vocab_start.contiguous() if use_ptr else None
        vocab_start_const = 0 if use_ptr else int(vocab_start)
        args = (
            drafts,
            probabilities,
            valid,
            tile_masses,
            copy_active.contiguous(),
            copy_tokens.contiguous(),
            vocab_start_ptr,
        )
        kwargs = {
            "VOCAB_START_CONST": vocab_start_const,
            "USE_VOCAB_START_PTR": use_ptr,
            "LOCAL_VOCAB": local_vocab,
            "BLOCKS": blocks,
            "TILE_BLOCK": tile_block,
            "HAS_MASSES": tile_masses is not None,
            "num_warps": 4,
        }
        grid = (blocks, batch * depth)
        if torch.compiler.is_compiling() or get_proxy_mode() is not None:
            torch.library.wrap_triton(_apply_copy_proposals_kernel)[grid](*args, **kwargs)
        else:
            _apply_copy_proposals_kernel[grid](*args, **kwargs)
        return _copy_offered_mask(copy_active)

    drafts.copy_(torch.where(copy_active, copy_tokens, drafts))
    valid.copy_(torch.where(copy_active, True, valid))
    local_ids = torch.arange(local_vocab, device=device, dtype=torch.int64) + vocab_start
    point_mass = (local_ids[None, None, :] == copy_tokens[:, :, None]).to(probabilities.dtype)
    probabilities.copy_(torch.where(copy_active[:, :, None], point_mass, probabilities))
    if tile_masses is not None:
        mass_blocks = tile_masses.shape[-1]
        local_target = copy_tokens - vocab_start
        in_shard = (local_target >= 0) & (local_target < local_vocab)
        target_tile = torch.where(in_shard, local_target // tile_block, -1)
        tile_ids = torch.arange(mass_blocks, device=device, dtype=torch.int64)
        tile_point_mass = (tile_ids[None, None, :] == target_tile[:, :, None]).to(tile_masses.dtype)
        tile_masses.copy_(torch.where(copy_active[:, :, None], tile_point_mass, tile_masses))
    return _copy_offered_mask(copy_active)


class PreparedParallelDraftRound:
    """Prepare verification and optionally its draft generator.

    A target-only forward receives proposals from its enclosing captured round;
    it allocates no draft decoder, Markov projection, or proposal samplers.
    """

    def __init__(self, owner, batch, width, name, *, generate_proposals=True, proposal_only=False,
                 target_token_capacity=None, forwards=None, verification_dummy=None):
        self.owner, self.batch, self.width = owner, batch, width
        self.vocabulary_parallel = getattr(owner, "vocabulary_parallel_sampling", False)
        self.name = name
        self.verification_dummy = verification_dummy
        tokens = batch * width if target_token_capacity is None else target_token_capacity
        if (type(tokens) is not int or not batch <= tokens <= batch * width
                or (proposal_only and target_token_capacity is not None)):
            raise ValueError("packed verification capacity must cover the request batch within its prefix bounds")
        self.target_token_capacity = tokens
        self.generate_proposals = generate_proposals
        self.proposal_only = proposal_only
        o, b, w = owner, batch, width
        device, blocks = o.pool.backing.device, o.inputs.page_tables.shape[1]
        forwards = o.forwards if forwards is None else forwards
        self.rows = torch.arange(b, device=device)
        self.draft_positions = torch.arange(w - 1, device=device) if self.vocabulary_parallel else None
        self.metadata = Metadata.allocate(tokens, b, blocks, device, samples=b * w, max_query_tokens=w)
        self.validator = prepare_batch(
            o.inputs,
            o.requests,
            Metadata.allocate(b, b, blocks, device),
            mode="decode",
            shard=o.weights.shard,
            pool=o.pool,
            verification_width=w,
            target_token_capacity=tokens,
            page_tokens=o.page_tokens,
        )
        self.validation = PreparedAllReduce(
            self.validator.validation_input,
            self.validator.validation_sum,
            runtime=o.collectives["runtime"],
            coordinator=o.collectives["coordinator"],
            session=o.session,
            name=name + ".validation",
        )
        self.decoder = None if proposal_only else forwards.decoder(
            o.weights,
            o.tables,
            o.scratch.rows(tokens, b * w),
            self.metadata,
            o.pool,
            mode="decode" if w == 1 else "verify",
            session=o.session,
            name=name + ".target",
            workspace=o.workspace,
            **o.collectives,
            **({"vocabulary_parallel": True} if self.vocabulary_parallel else {}),
        )
        self.context = None if proposal_only else forwards.context(
            o.weights,
            self.decoder.auxiliary,
            self.metadata,
            o.pool,
            session=o.session,
            name=name + ".context",
            workspace=o.workspace,
        )
        block = o.block_size
        self.draft_metadata = Metadata.allocate(b * block, b, blocks, device, samples=b * block)
        self.draft = (
            None
            if w == 1 or not generate_proposals
            else forwards.draft(
                o.weights,
                o.draft_scratch.rows(b * block, b * block),
                self.draft_metadata,
                o.pool,
                session=o.session,
                name=name + ".draft",
                workspace=o.workspace,
                **o.collectives,
                **({"vocabulary_parallel": True} if self.vocabulary_parallel else {}),
            )
        )
        self.selector = getattr(self.draft, "selector", None)
        self.draft_inputs = torch.full(
            (b, block), o.weights.config.draft.noise_token, dtype=torch.int64, device=device
        )
        self.markov_token = torch.zeros(b, dtype=torch.int64, device=device)
        self.markov = (
            None
            if self.draft is None or o.forwards.markov is None or self.vocabulary_parallel
            else o.forwards.markov(
                o.weights,
                self.markov_token,
                runtime=o.markov_runtime,
                coordinator=o.collectives["coordinator"],
                session=o.session,
                name=name + ".markov",
            )
        )
        self.recurrent = getattr(self.draft, "recurrent", None)
        if self.draft is not None and self.vocabulary_parallel and self.recurrent is None:
            from .dspark import PreparedMarkovEmbedding

            self.markov = PreparedMarkovEmbedding(o.weights, self.markov_token,
                                                  session=o.session, name=name + ".markov")
        self.shortlisted_selector = (self.selector if self.markov is None
            and callable(getattr(self.selector, "prepare_shortlists", None)) else None)
        self.before = torch.zeros(b, dtype=torch.int64, device=device)
        self.active = torch.zeros(b, dtype=torch.bool, device=device)
        self.candidates = torch.zeros((b, w), dtype=torch.int64, device=device)
        self.drafts = torch.zeros((b, w - 1), dtype=torch.int64, device=device)

        local_vocab = o.scratch.local_logits.shape[-1] if self.vocabulary_parallel else o.requests.vocab_size
        self.local_vocab_start = persistent_copy(torch.tensor(
            [o.weights.shard.rank * local_vocab if self.vocabulary_parallel else 0],
            dtype=torch.int64,
            device=device,
        ))
        self.probability_shape = (b, w - 1, local_vocab)
        self.draft_mass_shape = (b, w - 1, (local_vocab + 255) // 256)
        self.target_mass_shape = (b, w, (local_vocab + 255) // 256)
        self.logits = (torch.zeros((b, 1, local_vocab), dtype=torch.float32, device=device)
                       if generate_proposals and not self.vocabulary_parallel else None)
        self.probabilities = torch.zeros(self.probability_shape, dtype=torch.float32, device=device)
        self.draft_tile_masses = (torch.zeros(self.draft_mass_shape, dtype=torch.float32, device=device)
                                  if self.vocabulary_parallel else None)
        self.distribution_confidence = (self.vocabulary_parallel and self.draft is not None
                                        and getattr(o.weights.draft, "confidence_head", None) is None)
        self.draft_confidence = (torch.zeros((b, w - 1), dtype=torch.float32, device=device)
                                 if self.distribution_confidence else None)
        self.valid = torch.zeros((b, w - 1), dtype=torch.bool, device=device)
        self.seed, self.offset, self.base_offset, self.sampled = (
            torch.zeros_like(self.before) for _ in range(4)
        )
        self.operations = None
        self.proposal_block = None
        self.full_proposal = None

    def prepare(self):
        if self.operations is not None:
            return
        o, b, w = self.owner, self.batch, self.width
        if self.vocabulary_parallel:
            self._prepare_vocabulary_parallel()
            return
        args = (o.requests, o.inputs.row_slot_ids[:b], o.inputs.row_generations[:b], o.inputs.live_rows)
        target = None if self.proposal_only else PreparedSpeculativeDistributions(
            self.decoder.scratch.logits.view(b, w, -1), *args, self.drafts,
            token_limits=o.inputs.query_lengths[:b],
            workspace=None
        )
        head = getattr(getattr(o.weights, "draft", None), "markov_head", None)
        from .markov_sampling import RowScaledFP8MarkovWeight

        # Full-vocabulary transition fusion shares transition weight loads across b=1,2,4 and 8-row batches.
        fuse_markov = ((b in (1, 2, 4) or (b >= 8 and b % 8 == 0)) and w > 1 and self.generate_proposals
            and getattr(o, "block_proposal_broadcast", False) and o.markov_runtime is None
            and self.selector is None and self.markov is not None
            and ((isinstance(head, torch.Tensor) and head.dtype == torch.bfloat16 and head.is_contiguous())
                 or isinstance(head, RowScaledFP8MarkovWeight)))
        proposals = tuple(
            PreparedDraftDistribution(
                self.logits,
                *args,
                self.drafts,
                prefix_offset=i,
                seed=self.seed,
                offset=self.offset,
                sampled=self.sampled,
                workspace=None,
            )
            for i in range(w - 1 if self.generate_proposals and not fuse_markov else 0)
        )
        if target is not None:
            target.prepare()
        draft_logits_example = None
        shortlist_example = None
        shortlist_scores_example = None
        direct_shortlist = (self.shortlisted_selector is not None
                            and self.shortlisted_selector.top_k <= DRAFT_CANDIDATES
                            and callable(getattr(self.shortlisted_selector, "score_shortlist", None)))
        if self.draft is not None and (proposals or fuse_markov):
            if getattr(self.draft, "shortlist_output", False):
                if not direct_shortlist or fuse_markov:
                    raise ValueError("shortlist-only draft output requires direct shortlist proposal scoring")
            else:
                draft_logits_example = torch.empty(self.draft.logit_shape, dtype=self.draft.logit_dtype,
                                                   device=self.owner.requests.histogram.device)
            if self.shortlisted_selector is not None:
                device = self.owner.requests.histogram.device
                shortlist_example = torch.empty((b, self.draft_inputs.shape[1], self.shortlisted_selector.top_k),
                                                dtype=torch.int64, device=device)
                if direct_shortlist:
                    shortlist_scores_example = torch.empty(
                        (b, self.shortlisted_selector.top_k),
                        dtype=torch.float32, device=device)
        for index, proposal in enumerate(proposals):
            bias = (None if self.markov is None else torch.empty((b, o.requests.vocab_size),
                dtype=torch.float32, device=self.owner.requests.histogram.device))
            if direct_shortlist:
                proposal.bind_shortlist_scores(shortlist_scores_example,
                                               shortlist_example[:, index + o.logit_offset])
            elif self.shortlisted_selector is not None and self.shortlisted_selector.top_k <= DRAFT_CANDIDATES:
                proposal.bind_shortlist(shortlist_example[:, index + o.logit_offset])
            elif (self.draft is not None and self.selector is None
                    and (self.markov is None or bias is not None)):
                block = self.draft_inputs.shape[1]
                proposal.bind_logits(draft_logits_example.view(b, block, -1)[:, index + o.logit_offset],
                    bias)
            proposal.prepare()
        if fuse_markov:
            from .markov_sampling import PreparedMarkovProposals, PreparedDraftTokenBroadcast

            base = draft_logits_example.view(b, -1, draft_logits_example.shape[-1])[:, o.logit_offset:o.logit_offset + w - 1]
            self.full_proposal = PreparedMarkovProposals(base, o.weights.draft.markov_embedding, head,
                self.markov_token, self.drafts, self.probabilities, self.valid, o.requests,
                o.inputs.row_slot_ids[:b], o.inputs.row_generations[:b], o.inputs.live_rows,
                self.seed, self.base_offset, workspace=None,
                proposal_capacity=min(w - 1, self.target_token_capacity - 1))
            self.full_proposal.prepare()
            runtime = getattr(o, "result_runtime", None)
            if runtime is None:
                runtime = o.collectives["vocabulary_runtime"]
            self.proposal_block = PreparedDraftTokenBroadcast(self.drafts, self.valid, self.sampled,
                runtime=runtime, session=o.session, coordinator=o.collectives["coordinator"],
                name=self.name + ".proposal_tokens")
        broadcast_proposals = proposals
        if proposals and getattr(o, "block_proposal_broadcast", False):
            if o.markov_runtime is not None:
                raise ValueError("block proposal broadcast requires replicated proposal work")
            runtime = getattr(o, "result_runtime", None)
            if runtime is None:
                runtime = o.collectives["vocabulary_runtime"]
            self.proposal_block = PreparedDraftBlockBroadcast(
                proposals, self.drafts, self.probabilities, self.valid, self.sampled,
                runtime=runtime, session=o.session, coordinator=o.collectives["coordinator"],
                name=self.name + ".proposal_block")
            broadcast_proposals = ()
        if self.proposal_only:
            prepare_sampling_broadcasts(broadcast_proposals, None, collectives=o.collectives, session=o.session,
                                        name=self.name,
                                        result_runtime=getattr(o, "result_runtime", None))
            self.operations = None, proposals, None, None
            self._release_examples()
            return
        verify = PreparedStochasticVerification(
            self.probabilities,
            self.drafts,
            target.probabilities,
            *args,
            rng=o.rng,
            draft_valid=self.valid,
            target_valid=target.valid,
            token_limits=o.inputs.query_lengths[:b],
            workspace=None,
            dummy=self.verification_dummy,
        )
        verify.prepare()
        prepare_sampling_broadcasts(broadcast_proposals, verify, collectives=o.collectives, session=o.session,
            name=self.name, result_runtime=getattr(o, "result_runtime", None))
        from .speculation import PreparedSpeculativeStateCommit

        private = getattr(o, "study_width", None) is not None
        commit_type = PreparedSpeculativeStateCommit if private else PreparedSpeculativeCommit
        commit = commit_type(
            o.requests,
            verify.output,
            self.candidates,
            o.inputs.row_slot_ids[:b],
            o.inputs.row_generations[:b],
            o.inputs.live_rows,
            None if private else o.completion,
            o.frame,
            o.sequence,
            logits=None if private else self.decoder.scratch.logits.view(b, w, -1), rng=o.rng,
            draft_context_lag=0,
            token_limits=o.inputs.query_lengths[:b],
        )
        commit.prepare()
        if commit.logprobs is not None:
            commit.logprobs.release_logits()
        self.decoder.release_outputs()
        self.operations = target, proposals, verify, commit
        self._release_examples()

    def _release_examples(self):
        self.validator.functional()
        self.validation.functional(session=self.owner.session)
        if self.draft is not None:
            self.draft.release_outputs()
        if self.context is not None:
            self.context.functional()
        target, proposals, verify, commit = self.operations
        for operation in (target, *proposals, verify, commit):
            if operation is not None:
                operation.functional(session=self.owner.session)
        if self.full_proposal is not None:
            self.full_proposal.functional(session=self.owner.session)
        self.metadata = self.draft_metadata = None
        self.rows = self.draft_positions = None
        for name in ("draft_inputs", "markov_token", "before", "active", "candidates", "drafts",
                     "valid", "seed", "offset", "base_offset", "sampled", "logits", "probabilities",
                     "draft_tile_masses", "draft_confidence"):
            setattr(self, name, None)

    def _prepare_vocabulary_parallel(self):
        from .vocabulary_sampling import PreparedVocabularySoftmax, PreparedVocabularyBlockVerification

        o, b, w = self.owner, self.batch, self.width
        args = (o.requests, o.inputs.row_slot_ids[:b], o.inputs.row_generations[:b], o.inputs.live_rows)
        start = o.weights.shard.rank * self.probability_shape[-1]
        options = dict(start=start, runtime=o.sampling_runtime, session=o.session,
                       coordinator=o.collectives["coordinator"])
        if self.draft is not None:
            if getattr(self.draft, "recurrent_only", False):
                base = torch.empty((b, w - 1, self.probability_shape[-1]), dtype=self.draft.local_logit_dtype,
                                   device=o.requests.histogram.device)
            else:
                local = torch.empty(self.draft.local_logit_shape, dtype=self.draft.local_logit_dtype,
                                    device=o.requests.histogram.device).view(b, o.block_size, -1)
                base = local[:, o.logit_offset:o.logit_offset + w - 1]
            transition = ({} if self.recurrent is not None else dict(
                embedding=o.weights.draft.markov_embedding, head=o.weights.draft.markov_head))
            copy_options = (
                dict(
                    copy_active=torch.zeros((b, w - 1), dtype=torch.bool, device=o.requests.histogram.device),
                    copy_tokens=torch.zeros((b, w - 1), dtype=torch.int64, device=o.requests.histogram.device),
                )
                if getattr(o, "copy_ids", None) is not None
                else {}
            )
            self.full_proposal = PreparedVocabularySoftmax(base, self.probabilities, self.valid, *args,
                self.drafts, **options, name=self.name + ".proposal", workspace=None,
                **transition,
                pending=self.markov_token, seeds=self.seed, offsets=self.base_offset, tokens=self.drafts,
                tile_masses=self.draft_tile_masses, token_limits=o.inputs.query_lengths[:b],
                confidence=self.draft_confidence, **copy_options)
            self.full_proposal.prepare()
        if self.proposal_only:
            self.operations = None, (), None, None
            self._release_examples()
            return
        logits = self.decoder.local_logits.view(b, w, -1)
        from ignition.preparation.scratch import bind_scratch

        retained = bind_scratch(dict(probabilities=(tuple(logits.shape), torch.float32),
            valid=((b, w), torch.bool),
            tile_masses=(self.target_mass_shape, torch.float32)), device=logits.device, workspace=None)
        target = PreparedVocabularySoftmax(logits, retained["probabilities"], retained["valid"], *args,
            self.drafts, **options, name=self.name + ".target", workspace=None,
            tile_masses=retained["tile_masses"], token_limits=o.inputs.query_lengths[:b])
        target.prepare()
        verify = PreparedVocabularyBlockVerification(self.probabilities, self.drafts, target.probabilities, *args,
            **options, name=self.name + ".verification", rng=o.rng, draft_valid=self.valid,
            target_valid=target.valid, token_limits=o.inputs.query_lengths[:b], workspace=None,
            draft_tile_masses=self.draft_tile_masses, target_tile_masses=retained["tile_masses"])
        verify.prepare()
        from .vocabulary_logprobs import PreparedVocabularyParallelLogprobs

        logprobs = PreparedVocabularyParallelLogprobs(logits, o.completion,
            o.inputs.row_slot_ids[:b], o.inputs.row_generations[:b], o.inputs.live_rows, o.frame,
            vocabulary=o.requests.vocab_size, offset=start, runtime=o.logprob_runtime, session=o.session,
            coordinator=o.collectives["coordinator"], name=self.name + ".logprobs")
        commit = PreparedSpeculativeCommit(o.requests, verify.output, self.candidates,
            o.inputs.row_slot_ids[:b], o.inputs.row_generations[:b], o.inputs.live_rows,
            o.completion, o.frame, o.sequence, rng=o.rng, draft_context_lag=0,
            token_limits=o.inputs.query_lengths[:b], logprobs=logprobs)
        commit.prepare()
        self.operations = target, (), verify, commit
        self.decoder.release_outputs()
        self._release_examples()

    def _round(self):
        probabilities = torch.empty(self.probability_shape, dtype=torch.float32,
                                    device=self.owner.requests.histogram.device)
        proposal = self.propose(probabilities=probabilities)
        result = self.verify(probabilities=probabilities, proposal=proposal)
        return proposal, result

    def __call__(self):
        self._round()

    def observe(self):
        """Execute one round and return its metadata and acceptance for diagnostics."""
        proposal, result = self._round()
        metadata, acceptance, committed = result
        draft_metadata = proposal.metadata
        values = {}
        for prefix, item in (("target", metadata), ("draft", draft_metadata)):
            if item is not None:
                for field in ("live_counts", "starts", "lengths", "query_start_loc"):
                    values[f"{prefix}.{field}"] = getattr(item, field)
        for field in ("offered_drafts", "accepted_drafts", "survival", "valid"):
            values[f"acceptance.{field}"] = getattr(acceptance, field)
        for field in ("accepted_drafts", "emitted_count", "emitted_ids"):
            values[f"commit.{field}"] = getattr(committed, field)
        return values

    def propose(self, advance_rng=True, allowed=None, *, probabilities):
        if not self.generate_proposals:
            raise RuntimeError("verification-only preparation cannot generate proposals")
        o, b, w = self.owner, self.batch, self.width
        _, proposals, _, _ = self.operations
        validation_input, present = self.validator.validate()
        validated = self.validator.begin(self.validation(validation_input), present=present)
        device = o.requests.histogram.device
        active = validated.lengths > 0
        offsets = row_rotary_offsets(validated)
        if allowed is not None:
            active = active & allowed
        slots = o.inputs.row_slot_ids[:b].long()
        safe = slots.clamp(0, o.requests.capacity - 1)
        before = o.requests.consumed_tokens[safe]
        limits = torch.where(active, o.inputs.query_lengths[:b], 0)
        states = o.inputs.request_state_rows[safe, 0]
        candidates = torch.zeros((b, w), dtype=torch.int64, device=device)
        drafts = torch.zeros((b, w - 1), dtype=torch.int64, device=device)
        valid = torch.zeros((b, w - 1), dtype=torch.bool, device=device)
        candidates[:, 0].copy_(o.requests.pending_token[safe])
        base_offset = torch.where(o.generation[safe] == o.requests.generation[safe], o.offset[safe], 0)
        seed = o.requests.seed.view(torch.int64)[safe] ^ DRAFT_RNG_DOMAIN
        sampled = torch.zeros(b, dtype=torch.int64, device=device)
        draft_metadata = features = copy_offered_mask = None
        draft_tile_masses = (torch.empty(self.draft_mass_shape, dtype=torch.float32, device=device)
                             if self.vocabulary_parallel else None)
        draft_confidence = (torch.zeros((b, w - 1), dtype=torch.float32, device=device)
                            if getattr(self, "distribution_confidence", False) else None)
        if self.draft is not None:
            copy_active, copy_tokens = _resolve_copy_proposals(o, active, safe, before, limits, w - 1)
            block = o.block_size
            draft_inputs = torch.full((b, block), o.weights.config.draft.noise_token, dtype=torch.int64, device=device)
            draft_inputs[:, 0].copy_(candidates[:, 0])
            draft_metadata = Metadata.pack(before, active.int() * block, states,
                o.inputs.page_tables[safe], draft_inputs, o.inputs.live_rows,
                token_capacity=b * block, sample_count=b * block, rotary_offset=offsets)
            result = self.draft(draft_metadata)
            draft_output, selector_features = result[:2]
            features = result[2] if len(result) == 3 else None
            unary = shortlist_ids = draft_logits = None
            recurrent_only = getattr(self.draft, "recurrent_only", False)
            if getattr(self.draft, "shortlist_output", False):
                unary, shortlist_ids = draft_output
            elif not recurrent_only:
                draft_logits = draft_output.view(b, -1, draft_output.shape[-1])
                if self.shortlisted_selector is not None:
                    unary, shortlist_ids = self.shortlisted_selector.prepare_shortlists(draft_logits, features=selector_features)
            markov_token = torch.where(active, candidates[:, 0], 0)
            block_state = None
            copy_applied = False
            if self.full_proposal is not None:
                conf_kw = {"confidence": draft_confidence} if self.distribution_confidence else {}
                fused_copy = copy_active is not None and getattr(self.full_proposal, "has_copy", False)
                if self.vocabulary_parallel and self.recurrent is not None:
                    hidden = features["hidden"].view(b, block, -1)
                    context = features["recurrent_context"]
                    state = self.recurrent.begin(context)
                    if recurrent_only:
                        base = None
                        scores = torch.empty((b, w - 1, self.probability_shape[-1]),
                            dtype=self.draft.local_logit_dtype, device=device)
                    else:
                        base = draft_logits[:, o.logit_offset:o.logit_offset + w - 1]
                        scores = torch.empty_like(draft_logits)[:, o.logit_offset:o.logit_offset + w - 1]
                    copy_kwargs = dict(copy_active=copy_active, copy_tokens=copy_tokens) if fused_copy else {}
                    for index in range(w - 1):
                        state, logits = self.recurrent.step(state, markov_token,
                            hidden[:, index + o.logit_offset].contiguous(), context,
                            None if base is None else base[:, index])
                        scores[:, index].copy_(logits)
                        self.full_proposal.run_step(index, scores, drafts, pending=markov_token,
                            seeds=seed, offsets=base_offset, probabilities=probabilities, valid=valid,
                            tokens=drafts, tile_masses=draft_tile_masses, token_limits=limits,
                            **conf_kw, **copy_kwargs)
                        if copy_active is not None and not fused_copy:
                            drafts[:, index].copy_(torch.where(copy_active[:, index], copy_tokens[:, index], drafts[:, index]))
                        markov_token = drafts[:, index].clone()
                    copy_applied = fused_copy
                elif self.vocabulary_parallel:
                    base = draft_logits[:, o.logit_offset:o.logit_offset + w - 1]
                    if fused_copy:
                        self.full_proposal.run(base, drafts, pending=markov_token, seeds=seed, offsets=base_offset,
                            probabilities=probabilities, valid=valid, tokens=drafts,
                            tile_masses=draft_tile_masses, token_limits=limits,
                            copy_active=copy_active, copy_tokens=copy_tokens, **conf_kw)
                        copy_applied = True
                    elif copy_active is not None:
                        for index in range(w - 1):
                            self.full_proposal.run_step(index, base, drafts, pending=markov_token,
                                seeds=seed, offsets=base_offset, probabilities=probabilities, valid=valid,
                                tokens=drafts, tile_masses=draft_tile_masses, token_limits=limits, **conf_kw)
                            drafts[:, index].copy_(torch.where(copy_active[:, index], copy_tokens[:, index], drafts[:, index]))
                    else:
                        self.full_proposal.run(base, drafts, pending=markov_token, seeds=seed, offsets=base_offset,
                            probabilities=probabilities, valid=valid, tokens=drafts,
                            tile_masses=draft_tile_masses, token_limits=limits, **conf_kw)
                else:
                    base = draft_logits[:, o.logit_offset:o.logit_offset + w - 1]
                    self.full_proposal.run(base, markov_token, drafts, probabilities, valid, seed, base_offset)
                positions = torch.arange(w - 1, device=device)
                valid = valid & active[:, None] & (positions[None, :] < limits[:, None] - 1)
                sampled = drafts[:, -1]
                markov_token = sampled
            elif self.proposal_block is not None:
                block_state = self.proposal_block.begin(device)
            for i, sampler in enumerate(proposals):
                position = i + o.logit_offset
                bias = self.markov(markov_token) if self.markov is not None else None
                shortlist_scores = None
                source_logits = None if draft_logits is None else draft_logits[:, position]
                if getattr(sampler, "has_shortlist_scores", False):
                    proposal_logits = None
                    shortlist_scores = self.shortlisted_selector.score_shortlist(
                        markov_token, position, selector_features, unary, shortlist_ids)
                elif sampler.has_source_logits:
                    proposal_logits = torch.empty((b, 1, o.requests.vocab_size), dtype=torch.float32, device=device)
                else:
                    proposal_logits = source_logits.float()
                    if bias is not None:
                        proposal_logits = proposal_logits + bias
                    if self.shortlisted_selector is not None:
                        proposal_logits = self.shortlisted_selector.apply_shortlist(
                            proposal_logits, markov_token, position, selector_features, unary, shortlist_ids)
                    elif self.selector is not None:
                        proposal_logits = self.selector.apply(proposal_logits, markov_token, position, selector_features)
                    proposal_logits = proposal_logits.contiguous().view(b, 1, -1)
                scratch = {name: torch.empty(shape, dtype=dtype, device=device)
                           for name, (shape, dtype) in sampler.layout.items()}
                if block_state is not None:
                    scratch["support_ids"] = block_state["support_ids"][i]
                    scratch["support_probabilities"] = block_state["support_probabilities"][i]
                proposal_probabilities = sampler.run(scratch, proposal_logits,
                    source_logits=source_logits if sampler.has_source_logits else None,
                    shortlist_ids=None if shortlist_ids is None else shortlist_ids[:, position],
                    candidates=drafts, seed=seed, offset=base_offset + 4 * i, sampled=sampled,
                    **({"shortlist_scores": shortlist_scores} if shortlist_scores is not None else {}),
                    **({"source_bias": bias} if sampler.has_source_logits and bias is not None else {}))
                chosen = sampled if copy_active is None else torch.where(copy_active[:, i], copy_tokens[:, i], sampled)
                drafts[:, i].copy_(chosen)
                if block_state is None:
                    probabilities[:, i].copy_(proposal_probabilities[:, 0])
                    valid[:, i].copy_(scratch["valid"][:, 0] & active & (i < limits - 1))
                else:
                    self.proposal_block.retain(block_state, i, scratch, chosen)
                markov_token = chosen.clone()
            if self.proposal_block is not None:
                if self.full_proposal is not None:
                    drafts, valid, sampled = self.proposal_block.finish(drafts, valid, active, limits)
                else:
                    self.proposal_block.finish(block_state, drafts, probabilities, valid, sampled, active, limits)
            if copy_active is not None:
                if copy_applied:
                    copy_offered_mask = _copy_offered_mask(copy_active)
                else:
                    vocab_start = getattr(self, "local_vocab_start", None)
                    if vocab_start is None:
                        vocab_start = o.weights.shard.rank * probabilities.shape[-1] if self.vocabulary_parallel else 0
                    copy_offered_mask = _apply_copy_proposals(
                        drafts,
                        probabilities,
                        valid,
                        copy_active,
                        copy_tokens,
                        vocab_start=vocab_start,
                        tile_masses=draft_tile_masses,
                    )
                    if draft_confidence is not None:
                        draft_confidence.copy_(torch.where(copy_active, 1.0, draft_confidence))
        candidates[:, 1:].copy_(drafts)
        proposal = DraftProposalState(before, active, candidates, drafts, valid, base_offset, draft_metadata, features,
            offsets, draft_tile_masses, copy_offered_mask=copy_offered_mask, confidence=draft_confidence)
        if advance_rng and self.draft is not None:
            self.commit_draft_rng(proposal=proposal)
        return proposal

    def commit_draft_rng(self, drafts=None, *, proposal):
        o, b, w = self.owner, self.batch, self.width
        drafts = w - 1 if drafts is None else drafts
        slots = o.inputs.row_slot_ids[:b].long()
        slot_ids = torch.arange(o.requests.capacity, device=slots.device)
        matches = proposal.active[None, :] & (slots[None, :] == slot_ids[:, None])
        base = torch.where(matches, proposal.base_offset[None, :], 0).sum(1)
        o.offset.copy_(torch.where(matches.any(1), base + 4 * drafts, o.offset))
        o.generation.copy_(torch.where(matches.any(1), o.requests.generation, o.generation))

    def verify(self, proposal_valid=None, *, probabilities, proposal, token_limits=None):
        o, b, w = self.owner, self.batch, self.width
        target, _, verify, commit = self.operations
        slots = o.inputs.row_slot_ids[:b].long()
        safe = slots.clamp(0, o.requests.capacity - 1)
        active = proposal.active if proposal_valid is None else proposal.active & proposal_valid
        query_limits = o.inputs.query_lengths[:b] if token_limits is None else token_limits
        limits = torch.where(active, query_limits, 0)
        states = o.inputs.request_state_rows[safe, 0]
        metadata = Metadata.pack(proposal.before, limits, states, o.inputs.page_tables[safe],
            proposal.candidates, o.inputs.live_rows, token_capacity=self.target_token_capacity,
            sample_count=b * w, max_query_tokens=w, rotary_offset=proposal.rotary_offsets)
        hidden, logits, auxiliary, records = self.decoder(metadata)
        limit_kwargs = {} if token_limits is None else {"token_limits": query_limits}
        if self.vocabulary_parallel:
            target_tile_masses = torch.empty(self.target_mass_shape, dtype=torch.float32, device=logits.device)
            target_probabilities, target_valid, _ = target.run(logits.view(b, w, -1), proposal.drafts,
                tile_masses=target_tile_masses, token_limits=query_limits)
            if proposal.tile_masses is None:
                if w > 1:
                    raise ValueError(
                        "vocabulary-parallel verification requires proposal.tile_masses when width > 1"
                    )
                draft_masses = torch.empty(self.draft_mass_shape, dtype=torch.float32, device=logits.device)
            else:
                draft_masses = proposal.tile_masses
            acceptance = verify.run(probabilities, proposal.drafts, target_probabilities, proposal.valid, target_valid,
                token_limits=query_limits, draft_tile_masses=draft_masses, target_tile_masses=target_tile_masses)
        else:
            target_scratch = {name: torch.empty(shape, dtype=dtype, device=logits.device)
                              for name, (shape, dtype) in target.layout.items()}
            verify_scratch = {name: torch.empty(shape, dtype=dtype, device=logits.device)
                              for name, (shape, dtype) in PreparedStochasticVerification.scratch_layout(
                                  b, w, o.requests.vocab_size, consume_inputs=True).items()}
            target.run(target_scratch, logits.view(b, w, -1), candidates=proposal.drafts, **limit_kwargs)
            acceptance = verify.run(verify_scratch, probabilities, target_scratch["probabilities"],
                target_scratch["valid"], disjoint=True, consume_inputs=True,
                drafts=proposal.drafts, draft_valid=proposal.valid, **limit_kwargs)
        if proposal_valid is not None:
            acceptance.valid.logical_and_(proposal_valid)
        commit_kwargs = dict(limit_kwargs)
        if proposal.copy_offered_mask is not None:
            offered = (limits - 1).clamp(0, max(0, w - 1)).to(torch.int64)
            offered_mask = torch.where(
                offered >= 63,
                (1 << 63) - 1,
                (torch.ones_like(offered) << offered.clamp_max(62)) - 1,
            )
            commit_kwargs["copy_offered_mask"] = proposal.copy_offered_mask & offered_mask
        committed = commit(logits.view(b, w, -1), acceptance=acceptance, inputs=proposal.candidates, **commit_kwargs)
        counts = committed.emitted_count
        if w > 1:
            self.decoder.commit(counts, records, metadata)
        rows = torch.arange(b, device=hidden.device)
        hidden = hidden.view(b, w, o.weights.config.text["hidden_size"])[rows, (counts - 1).long().clamp_min(0)]
        save_hidden(hidden, o.pool.last_hidden, metadata.state_rows, counts > 0)
        if self.context is not None:
            self.context(auxiliary, counts, metadata)
        return metadata, acceptance, committed


class PreparedCopyOnlySpeculativeRound:
    """Target-only wide copy-verify round that skips DSpark and Markov drafting.

    Host copy proposals are validated for slot liveness, generation, cursor
    freshness, unconstrained grammar, and contiguous in-vocabulary token IDs,
    then verified directly through the target decoder with point-mass ``q=1.0``
    proposals. Accepted target feedback updates ``PreparedContext`` so the next
    learned draft round retains exact draft context KV state.
    """

    def __init__(self, target, *, name=None):
        if (target.generate_proposals or target.proposal_only or target.width <= 1
                or target.draft is not None or target.markov is not None):
            raise ValueError("copy-only speculative round requires a target-only verification round with width > 1")
        copy_ids = getattr(target.owner, "copy_ids", None)
        if copy_ids is None or copy_ids.shape[1] < target.width - 1:
            raise ValueError("copy-only speculative round requires copy_token_ids covering width - 1 drafts")
        self.target = target
        self.name = target.name if name is None else name
        o = target.owner
        vocab_parallel = getattr(target, "vocabulary_parallel", False)
        local_vocab = target.probability_shape[-1] if vocab_parallel else 0
        self.local_vocab_start = persistent_copy(torch.tensor(
            [o.weights.shard.rank * local_vocab if vocab_parallel else 0],
            dtype=torch.int64,
            device=copy_ids.device,
        ))

    def _round(self):
        t = self.target
        o, b, w = t.owner, t.batch, t.width
        device = o.requests.histogram.device
        validation_input, present = t.validator.validate()
        validated = t.validator.begin(t.validation(validation_input), present=present)
        active = validated.lengths > 0
        offsets = row_rotary_offsets(validated)
        slots = o.inputs.row_slot_ids[:b].long()
        safe = slots.clamp(0, o.requests.capacity - 1)
        before = o.requests.consumed_tokens[safe]
        limits = torch.where(active, o.inputs.query_lengths[:b], 0)
        candidates = torch.zeros((b, w), dtype=torch.int64, device=device)
        candidates[:, 0].copy_(o.requests.pending_token[safe])
        drafts = torch.zeros((b, w - 1), dtype=torch.int64, device=device)
        valid = torch.zeros((b, w - 1), dtype=torch.bool, device=device)
        probabilities = torch.zeros(t.probability_shape, dtype=torch.float32, device=device)
        draft_tile_masses = (
            torch.zeros(t.draft_mass_shape, dtype=torch.float32, device=device)
            if t.vocabulary_parallel
            else None
        )
        copy_active, copy_tokens = _resolve_copy_proposals(o, active, safe, before, limits, w - 1)
        copy_offered_mask = _apply_copy_proposals(
            drafts,
            probabilities,
            valid,
            copy_active,
            copy_tokens,
            vocab_start=self.local_vocab_start if t.vocabulary_parallel else 0,
            tile_masses=draft_tile_masses,
        )
        candidates[:, 1:].copy_(drafts)
        effective_count = copy_active.to(torch.int32).sum(dim=1)
        effective_limits = torch.where(active, (effective_count + 1).clamp_max(limits), 0)
        base_offset = torch.zeros(b, dtype=torch.int64, device=device)
        proposal = DraftProposalState(
            before,
            active,
            candidates,
            drafts,
            valid,
            base_offset,
            None,
            None,
            offsets,
            draft_tile_masses,
            copy_offered_mask,
        )
        result = t.verify(
            probabilities=probabilities,
            proposal=proposal,
            token_limits=effective_limits,
        )
        return proposal, result

    def __call__(self):
        _, result = self._round()
        return result

    def observe(self):
        """Execute one copy-only round and return its metadata and acceptance for diagnostics."""
        _, result = self._round()
        metadata, acceptance, committed = result
        values = {}
        for field in ("live_counts", "starts", "lengths", "query_start_loc"):
            values[f"target.{field}"] = getattr(metadata, field)
        for field in ("offered_drafts", "accepted_drafts", "survival", "valid"):
            values[f"acceptance.{field}"] = getattr(acceptance, field)
        for field in ("accepted_drafts", "emitted_count", "emitted_ids"):
            values[f"commit.{field}"] = getattr(committed, field)
        return values
