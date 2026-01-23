.PHONY: sync-skills check-skills

# Skill sync: .claude/skills/ is source of truth
sync-skills:
	@echo "Syncing skills from .claude/skills/ to .codex/skills/ and skills/..."
	@mkdir -p .codex/skills/langfuse skills/langfuse
	@cp -R .claude/skills/langfuse/* .codex/skills/langfuse/
	@cp -R .claude/skills/langfuse/* skills/langfuse/
	@echo "✓ Skills synced"

check-skills:
	@echo "Checking skill sync..."
	@diff -rq .claude/skills/langfuse .codex/skills/langfuse || (echo "❌ .codex/skills/langfuse out of sync" && exit 1)
	@diff -rq .claude/skills/langfuse skills/langfuse || (echo "❌ skills/langfuse out of sync" && exit 1)
	@echo "✓ Skills in sync"
