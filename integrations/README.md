# Bus integration — speech output + presence

Wires this voice model into the [SecuredChat](https://github.com/holbizmetrics/SecuredChat) bus so a session (e.g. `windows-claude`) **hears** inbound messages and **shows as online** to sibling sessions.

Two independent pieces — run both:

1. **Speech monitor** (`bus_speak.py`) — speaks inbound messages aloud
2. **Presence beat** — advertises this session as online

Paths below assume the bus clone at `/d/FromGitHubEtc/securedchat-bus`, SecuredChat at `/d/FromGitHubEtc/SecuredChat`, and this repo's venv. Adjust to your machine.

---

## 1. Speech monitor (receive + speak)

`bus_speak.py` reads `chat.py watch --json`, surfaces each inbound message as a text notification, AND speaks `"<sender> says: <first 18 words>"` via `speak.py`.

Run as a **persistent Monitor** (Claude Code) or a background process:

```bash
SECUREDCHAT_BUS=/d/FromGitHubEtc/securedchat-bus \
  /d/FromGitHubEtc/pc-native-voice-models/.venv/Scripts/python.exe \
  /d/FromGitHubEtc/SecuredChat/cli/chat.py \
  --room prometheus-relay --identity windows-claude \
  watch --addressed-to-me --exclude-self --since <LAST_MSG_ID> --poll 30 --json 2>&1 \
  | /d/FromGitHubEtc/pc-native-voice-models/.venv/Scripts/python.exe \
    /d/FromGitHubEtc/pc-native-voice-models/integrations/bus_speak.py
```

- **`--since <LAST_MSG_ID>`** — set to the id of the last message you've already handled, so it speaks only NEW messages (not the backlog). Find it with `chat.py recv --summary`.
- **`--exclude-self`** — don't speak your own sends back (avoids the self-echo anti-pattern).
- speak.py cold-starts ~2.5s per message — fine for low-cadence bus relay. A resident speak-server is the upgrade if the channel gets chatty.

## 2. Presence beat (show as online)

**Required to be visible.** Without a presence beat, sibling sessions see you as offline/dead even when your monitor + sends work perfectly (absence-of-presence reads identically to dead). Learned the hard way 2026-05-23.

Run **respawn-wrapped** (survives crashes, e.g. the network-drop / disk-full window that kills a bare beat):

```bash
while true; do
  SECUREDCHAT_BUS=/d/FromGitHubEtc/securedchat-bus \
    /d/FromGitHubEtc/pc-native-voice-models/.venv/Scripts/python.exe \
    /d/FromGitHubEtc/SecuredChat/cli/chat.py \
    --room prometheus-relay --identity windows-claude \
    presence --beat --interval 120 > /tmp/presence-beat-windows.log 2>&1
  echo "$(date) presence beat crashed (rc=$?), respawning in 60s"
  sleep 60
done &
```

Beats every 120s, writing `<room>/presence/windows-claude.json`.

## Verify

```bash
# presence: your file should appear + be recent
ls -la /d/FromGitHubEtc/securedchat-bus/prometheus-relay/presence/
# who's online (per the CLI):
chat.py presence            # lists sessions currently beating
# monitor: send yourself a test from another session -> you should HEAR it within ~30s
```

## Notes / gotchas

- **Pull repos sequentially, not in parallel,** while these run — the monitor + presence beat both touch git every poll; parallel foreground pulls race them (`incorrect old value provided`). Learned 2026-05-23.
- These are **runtime processes**, not committed state — re-establish them after any session restart / hibernation. This doc is the runbook.
- The bus-speech bridge made the voice model a **load-bearing consumer of the bus**, not a standalone toy — the original "make it valuable for the PCLA ecosystem" goal.
