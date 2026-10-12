"""Long-term memory: write policy, de-duplication, conflicts, retrieval, isolation."""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import select

from app.db.enums import MemoryKind, MemoryProvenance, MemoryStatus
from app.db.models import Memory, MemoryAudit, User
from app.db.repositories import memory as memory_repo
from app.memory.service import MemoryService
from app.schemas.agent import MemoryCandidate

from .fakes import blend, one_hot

pytestmark = pytest.mark.db


def candidate(
    content: str,
    *,
    kind: MemoryKind = MemoryKind.FEAR,
    provenance: MemoryProvenance = MemoryProvenance.AGENT_INFERRED,
    evidence: str = "",
) -> MemoryCandidate:
    return MemoryCandidate(
        kind=kind,
        content=content,
        normalized_key=content[:12],
        provenance=provenance,
        confidence=0.6,
        importance=4,
        evidence=evidence,
    )


async def _persist(database, settings, user_id, candidates, **kwargs):
    service = MemoryService(settings)
    async with database.session() as session:
        return await service.persist(
            session,
            user_id=user_id,
            session_id=kwargs.get("session_id"),
            message_id=None,
            candidates=candidates,
        )


async def _all_memories(database) -> list[Memory]:
    async with database.session() as session:
        return list((await session.execute(select(Memory).order_by(Memory.created_at))).scalars())


# ----------------------------------------------------------------- write policy
async def test_inferred_memory_stays_a_candidate(settings, database, user_id, patch_llm):
    patch_llm([[]])
    outcomes = await _persist(database, settings, user_id, [candidate("你害怕失去稳定的收入")])
    memory = outcomes[0].memory
    assert memory.status == str(MemoryStatus.CANDIDATE)
    assert memory.provenance == str(MemoryProvenance.AGENT_INFERRED)


async def test_user_stated_memory_is_active_immediately(settings, database, user_id, patch_llm):
    patch_llm([[]])
    outcomes = await _persist(
        database,
        settings,
        user_id,
        [
            candidate(
                "你的硬约束是每月房贷 8000 元不能断",
                kind=MemoryKind.CONSTRAINT,
                provenance=MemoryProvenance.USER_STATED,
                evidence="每个月房贷 8000，一天都不能断",
            )
        ],
    )
    memory = outcomes[0].memory
    assert memory.status == str(MemoryStatus.ACTIVE)
    assert memory.evidence


async def test_write_is_recorded_in_the_audit_trail(settings, database, user_id, patch_llm):
    patch_llm([[]])
    outcomes = await _persist(database, settings, user_id, [candidate("你想要被认可为专业人士")])
    async with database.session() as session:
        audits = list(
            (await session.execute(select(MemoryAudit).where(MemoryAudit.memory_id == outcomes[0].memory.id))).scalars()
        )
    assert [row.action for row in audits] == ["created"]
    assert audits[0].actor == "agent"


# ------------------------------------------------------------ de-duplication
async def test_same_memory_again_is_merged_not_duplicated(settings, database, user_id, patch_llm):
    install = patch_llm([[], []])
    install.embedder.set_vector("你害怕失去稳定的收入", one_hot(install.embedder.dim, 1))
    install.embedder.set_vector("你担心辞职之后就没有稳定收入了", one_hot(install.embedder.dim, 1))

    await _persist(database, settings, user_id, [candidate("你害怕失去稳定的收入")])
    outcomes = await _persist(database, settings, user_id, [candidate("你担心辞职之后就没有稳定收入了")])

    assert outcomes[0].action == "merged"
    rows = await _all_memories(database)
    assert len(rows) == 1
    assert rows[0].content == "你担心辞职之后就没有稳定收入了"  # the more specific wording wins

    async with database.session() as session:
        audits = list((await session.execute(select(MemoryAudit).where(MemoryAudit.memory_id == rows[0].id))).scalars())
    assert sorted(row.action for row in audits) == ["created", "merged"]


async def test_confirmation_upgrades_an_inferred_memory(settings, database, user_id, patch_llm):
    install = patch_llm([[], []])
    install.embedder.set_vector("你最怕的是失去可回退的路径", one_hot(install.embedder.dim, 2))
    install.embedder.set_vector("你最怕的是失去可以回头的路", one_hot(install.embedder.dim, 2))

    await _persist(database, settings, user_id, [candidate("你最怕的是失去可回退的路径")])
    outcomes = await _persist(
        database,
        settings,
        user_id,
        [
            candidate(
                "你最怕的是失去可以回头的路",
                provenance=MemoryProvenance.USER_CONFIRMED,
                evidence="对，就是这个",
            )
        ],
    )
    memory = outcomes[0].memory
    assert memory.provenance == str(MemoryProvenance.USER_CONFIRMED)
    assert memory.status == str(MemoryStatus.ACTIVE)


# ------------------------------------------------------------------- conflicts
async def test_contradiction_supersedes_the_old_memory(settings, database, user_id, patch_llm):
    # Only the second write needs the model: the first has no neighbours to judge.
    install = patch_llm([[{"verdicts": [{"index": 0, "verdict": "contradict", "note": "用户改变了决定"}]}]])
    old_text = "你倾向于留在现在的城市"
    new_text = "你已经决定搬去合伙人的城市"
    install.embedder.set_vector(old_text, one_hot(install.embedder.dim, 3))
    # cosine ≈ 0.80: similar enough to ask, below the merge threshold.
    install.embedder.set_vector(new_text, blend(install.embedder.dim, 3, 4, 0.8))

    await _persist(database, settings, user_id, [candidate(old_text)])
    outcomes = await _persist(database, settings, user_id, [candidate(new_text, kind=MemoryKind.FEAR)])

    assert outcomes[0].action == "superseded"
    rows = await _all_memories(database)
    assert len(rows) == 2
    old = next(row for row in rows if row.content == old_text)
    new = next(row for row in rows if row.content == new_text)
    assert old.status == str(MemoryStatus.SUPERSEDED)
    assert old.superseded_by == new.id


async def test_unrelated_neighbours_are_kept_side_by_side(settings, database, user_id, patch_llm):
    # Different kinds are never compared, so no judgement call is needed.
    install = patch_llm([[]])
    first = "你害怕失去稳定的收入"
    second = "你希望在四十岁前换一条职业路径"
    install.embedder.set_vector(first, one_hot(install.embedder.dim, 5))
    install.embedder.set_vector(second, blend(install.embedder.dim, 5, 6, 0.8))

    await _persist(database, settings, user_id, [candidate(first)])
    await _persist(database, settings, user_id, [candidate(second, kind=MemoryKind.GOAL)])

    kinds = {row.kind for row in await _all_memories(database)}
    # Different kinds never share a neighbour query, so both survive untouched.
    assert MemoryKind.FEAR.value in kinds


# ------------------------------------------------------------------ retrieval
async def test_retrieval_is_relevance_based(settings, database, user_id, patch_llm):
    install = patch_llm([[], []])
    relevant = "你怕的是三十五岁之后再也没人愿意给你机会"
    unrelated = "你喜欢在周末做木工"
    install.embedder.set_vector(relevant, one_hot(install.embedder.dim, 7))
    install.embedder.set_vector(unrelated, one_hot(install.embedder.dim, 8))

    await _persist(
        database,
        settings,
        user_id,
        [
            candidate(relevant, provenance=MemoryProvenance.USER_STATED, evidence="我怕过了三十五就没人要了"),
            candidate(unrelated, kind=MemoryKind.PREFERENCE, provenance=MemoryProvenance.USER_STATED),
        ],
    )

    install.embedder.set_vector("我在纠结要不要换工作，怕年龄是问题", one_hot(install.embedder.dim, 7))
    service = MemoryService(settings)
    async with database.session() as session:
        result = await service.recall(session, user_id=user_id, query="我在纠结要不要换工作，怕年龄是问题")
    assert [ref.content for ref in result.memories] == [relevant]
    assert result.memories[0].provenance == MemoryProvenance.USER_STATED


async def test_recall_marks_memories_as_used(settings, database, user_id, patch_llm):
    install = patch_llm([[]])
    text = "你害怕被同行超越"
    install.embedder.set_vector(text, one_hot(install.embedder.dim, 9))
    await _persist(database, settings, user_id, [candidate(text)])

    service = MemoryService(settings)
    async with database.session() as session:
        await service.recall(session, user_id=user_id, query=text)

    rows = await _all_memories(database)
    assert rows[0].use_count == 1
    assert rows[0].last_used_at is not None


async def test_memories_are_isolated_per_user(settings, database, user_id, patch_llm):
    install = patch_llm([[]])
    text = "你害怕失去稳定的收入"
    install.embedder.set_vector(text, one_hot(install.embedder.dim, 10))
    await _persist(database, settings, user_id, [candidate(text)])

    other_user = uuid.uuid4()
    async with database.session() as session:
        session.add(User(id=other_user, display_name="另一个人"))

    service = MemoryService(settings)
    async with database.session() as session:
        result = await service.recall(session, user_id=other_user, query=text)
    assert result.memories == []


async def test_deleted_memories_are_not_recalled(settings, database, user_id, patch_llm):
    install = patch_llm([[]])
    text = "你害怕失去稳定的收入"
    install.embedder.set_vector(text, one_hot(install.embedder.dim, 11))
    outcomes = await _persist(database, settings, user_id, [candidate(text)])

    from datetime import UTC, datetime

    async with database.session() as session:
        row = await memory_repo.get_memory(session, outcomes[0].memory.id, user_id=user_id)
        assert row is not None
        row.deleted_at = datetime.now(tz=UTC)

    service = MemoryService(settings)
    async with database.session() as session:
        result = await service.recall(session, user_id=user_id, query=text)
    assert result.memories == []

    async with database.session() as session:
        visible = await memory_repo.list_memories(session, user_id=user_id)
    assert visible == []


async def test_user_verdict_is_applied_and_audited(settings, database, user_id, patch_llm):
    from app.memory.consolidator import apply_user_verdict

    patch_llm([[]])
    outcomes = await _persist(database, settings, user_id, [candidate("你害怕的是失败本身")])
    memory_id = outcomes[0].memory.id

    async with database.session() as session:
        row = await memory_repo.get_memory(session, memory_id, user_id=user_id)
        assert row is not None
        await apply_user_verdict(
            session,
            memory=row,
            action="correct",
            corrected_text="我怕的不是失败，是失败以后没人再相信我",
        )

    async with database.session() as session:
        row = await memory_repo.get_memory(session, memory_id, user_id=user_id)
        assert row is not None
        assert row.content == "我怕的不是失败，是失败以后没人再相信我"
        assert row.provenance == str(MemoryProvenance.USER_STATED)
        assert row.status == str(MemoryStatus.ACTIVE)
        audits = await memory_repo.list_audit(session, memory_id=memory_id, user_id=user_id)
    assert audits[0].actor == "user"
