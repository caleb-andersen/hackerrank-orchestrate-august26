"""Consistency audit: do the published sample patterns hold on the full 110?

Read-only. Reports counts and row ids for three structural categories drawn from
labelled sample rows. Lives under code/evaluation/ because it opens
dataset/sample_messages.csv (AGENTS.md §9.8) and because it is not on the
decision path. Changes nothing.

Category A  (sample_msg_004): verified business + clean domain + a live
            why_user_knows_account -> the sample labels this notify.
Category B  (sample_msg_049): personal first contact, no prior history ->
            the sample labels this digest/unknown, not mute.
Category C  (no sample anchor): group admin sender + high user read rate,
            content is an operational notice.

Content judgement (is this actually benign / actually an operational notice)
is deliberately NOT automated here: the router's own scanners decided the
routing, so reusing them to define the category would hide exactly the
over-firing this audit is looking for. The script emits the candidate set with
raw text; the reading is done by hand.
"""

import csv
import os

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "dataset")

# why_user_knows_account values that describe a dormant or prospect-only
# relationship rather than a live one. sample_msg_004's anchor is
# "recent_grocery_delivery" -- a completed, recent transaction.
DORMANT_MARKERS = ("old_", "opted_out", "_search", "_interest", "watchlist", "ignored_")


def rd(name):
    with open(os.path.join(DATA, name), encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def is_active_relationship(why):
    return bool(why) and not any(m in why for m in DORMANT_MARKERS)


def short(text, n=150):
    return " ".join((text or "").split())[:n]


def main():
    messages = rd("messages.csv")
    preds = {r["message_id"]: r for r in rd("output.csv")}
    biz = {r["business_id"]: r for r in rd("business_accounts.csv")}
    ubh = {(r["user_id"], r["business_id"]): r for r in rd("user_business_history.csv")}
    groups = {r["group_id"]: r for r in rd("groups.csv")}
    members = {(r["group_id"], r["user_id"]): r for r in rd("group_members.csv")}
    users = {r["user_id"]: r for r in rd("users.csv")}
    history = rd("message_history.csv")

    prior_personal = set()
    for h in history:
        if h["sender_user_id"]:
            prior_personal.add((h["user_id"], h["sender_user_id"]))

    cat_a, cat_b, cat_c = [], [], []

    for m in messages:
        mid = m["message_id"]
        pred = preds.get(mid, {})
        action = pred.get("action", "MISSING")

        if m["conversation_type"] == "business" and m["business_id"] in biz:
            b = biz[m["business_id"]]
            rel = ubh.get((m["user_id"], m["business_id"]))
            why = rel["why_user_knows_account"] if rel else ""
            clean = b["domain_used_by_sender"] == b["official_domain"]
            if b["verified"] == "1" and clean and why:
                cat_a.append({
                    "message_id": mid,
                    "action": action,
                    "brand": b["brand_name"],
                    "why": why,
                    "active": is_active_relationship(why),
                    "opened_30d": rel["messages_opened_30d"],
                    "dismissed_30d": rel["messages_dismissed_30d"],
                    "allows_promos": rel["allows_promotions"],
                    "reason": short(pred.get("reason", ""), 110),
                    "text": short(m["message_text"], 200),
                    "media": m["media_type"],
                })

        if m["conversation_type"] == "personal" and m["sender_user_id"]:
            if (m["user_id"], m["sender_user_id"]) not in prior_personal:
                cat_b.append({
                    "message_id": mid,
                    "action": action,
                    "sender": m["sender_user_id"],
                    "reason": short(pred.get("reason", ""), 110),
                    "text": short(m["message_text"], 220),
                    "media": m["media_type"],
                    "fwd": m["forwarded_count"],
                })

        if m["conversation_type"] == "group" and m["sender_user_id"]:
            sender_row = members.get((m["group_id"], m["sender_user_id"]))
            user_row = members.get((m["group_id"], m["user_id"]))
            grp = groups.get(m["group_id"], {})
            if sender_row and sender_row["role"] == "admin":
                sent = int(grp.get("messages_30d") or 0)
                read = int(user_row["messages_read_30d"]) if user_row else 0
                grp_read_rate = (read / sent) if sent else 0.0
                u = users.get(m["user_id"], {})
                cat_c.append({
                    "message_id": mid,
                    "action": action,
                    "group": grp.get("group_name", ""),
                    "grp_read_rate": round(grp_read_rate, 2),
                    "grp_dismissed": user_row["notifications_dismissed_30d"] if user_row else "?",
                    "muted_by_user": user_row["group_muted_by_user"] if user_row else "?",
                    "user_opened_30d": u.get("messages_opened_30d", "?"),
                    "reason": short(pred.get("reason", ""), 110),
                    "text": short(m["message_text"], 200),
                    "media": m["media_type"],
                })

    for title, rows in (
        ("CATEGORY A - verified business, clean domain, known account", cat_a),
        ("CATEGORY B - personal first contact, no prior history", cat_b),
        ("CATEGORY C - group admin sender", cat_c),
    ):
        print("=" * 100)
        print(f"{title}  (n={len(rows)})")
        muted = [r for r in rows if r["action"] == "mute"]
        print(f"  muted: {len(muted)}   " + "  ".join(
            f"{a}={sum(1 for r in rows if r['action'] == a)}"
            for a in ("notify", "digest", "mute")))
        print("=" * 100)
        for r in rows:
            head = " | ".join(f"{k}={v}" for k, v in r.items()
                              if k not in ("text", "reason"))
            print(f"\n{head}")
            print(f"    reason : {r['reason']}")
            print(f"    text   : {r['text']}")


if __name__ == "__main__":
    main()
