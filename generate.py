"""Synthetic user generator (templates, so the answer key is exact).

Each user gets a timeline of sessions. A few sessions carry facts, and the rest are filler:
    1. intro      -> pet name, home city, work email
    2. sibling    -> sibling name + sibling city (another person's facts, a distractor)
    3. move       -> user moves to a new city and a new company   (updated fact + previous value)
    4. forget     -> user asks the assistant to forget the old work email   (deleted fact)
    filler        -> small talk with no personal facts, put between the fact sessions

Usage:
    python generate.py                         # 10 users, 20 sessions each -> data/generated/
    python generate.py --users 50 --sessions 40 --seed 7
"""
import argparse
import json
import os
import random

# ---------- value pools ----------
FIRST_NAMES = ["Arjun", "Meera", "Rohan", "Ananya", "Vikram", "Sneha", "Karthik", "Divya", "Aditya", "Kavya",
               "Rahul", "Pooja", "Nikhil", "Lakshmi", "Siddharth", "Riya", "Varun", "Nisha", "Harsha", "Tara"]
# The sibling's name must match the relation. "My sister Arun" confuses fact extraction
# (Zep lost the word "sister" in run 2), so names are split by relation.
SIBLING_NAMES = {
    "sister": ["Priya", "Asha", "Deepa", "Swathi", "Latha"],
    "brother": ["Ravi", "Manoj", "Arun", "Suresh", "Mahesh"],
    "cousin": ["Priya", "Asha", "Deepa", "Swathi", "Latha", "Ravi", "Manoj", "Arun", "Suresh", "Mahesh"],
}
PET_NAMES = ["Bruno", "Coco", "Simba", "Milo", "Luna", "Rocky", "Bella", "Oscar", "Toffee", "Leo", "Zara", "Biscuit"]
PET_KINDS = ["beagle", "labrador", "cat", "parrot", "golden retriever", "rabbit"]
CITIES = ["Hyderabad", "Bangalore", "Chennai", "Mumbai", "Pune", "Delhi", "Kolkata", "Kochi", "Jaipur", "Ahmedabad"]
COMPANIES = ["Razorpay", "Zomato", "Swiggy", "Flipkart", "Infosys", "Freshworks", "Zoho", "Cred", "Meesho", "Ola"]

# Topics that are never mentioned. The answer is always UNKNOWN.
NEVER_MENTIONED = [
    "What's my favorite football team?",
    "What's my blood type?",
    "What car do I drive?",
    "What's my favorite movie?",
    "What's my mother's name?",
    "What did I study in college?",
]

# ---------- message templates ----------
# Each template is (user message, assistant reply). Several versions per fact, so the text is not identical for every user.
INTRO = [
    ("Hey! Quick intro: I live in {city} and I just got a {kind} named {pet}.",
     "Nice to meet you! {pet} sounds adorable."),
    ("Hi, I'm new here. I'm based in {city}. Oh and my {kind} {pet} says hi.",
     "Hello! Say hi to {pet} from me."),
    ("Hello! I recently adopted a {kind}, his name is {pet}. I'm in {city} by the way.",
     "Congrats on adopting {pet}!"),
]
EMAIL = [
    ("For work stuff, my email is {email}, I work at {company}.",
     "Got it, I'll keep that in mind."),
    ("My work email is {email} if you need it. I'm at {company} right now.",
     "Thanks, noted."),
]
SIBLING = [
    ("My {rel} {sib} lives in {sib_city}. We talk every weekend.",
     "That's lovely, staying close matters."),
    ("Just got off a call with my {rel} {sib}, who lives in {sib_city}.",
     "Hope the call went well!"),
    ("Planning to visit my {rel} {sib} in {sib_city} next month.",
     "Sounds like a great trip."),
]
MOVE = [
    ("Big news: I moved from {old_city} to {new_city}! Starting a new job at {new_company}.",
     "Congratulations on the move and the new job!"),
    ("I finally relocated to {new_city}. Left {old_city} last week, joining {new_company} on Monday.",
     "That's a big change, good luck at {new_company}!"),
]
FORGET = [
    ("Please forget my old work email, I don't use it anymore.",
     "Okay, I've forgotten your old work email."),
    ("Can you delete the work email I gave you earlier? It's not valid anymore.",
     "Done, I've removed your old work email."),
]
# Filler: no personal facts. Some mention cities, to make retrieval harder (hard negatives).
FILLER = [
    ("It's raining like crazy today lol.", "Stay dry!"),
    ("Can you recommend a good book for a long flight?", "Try 'Project Hail Mary', it's a fun read."),
    ("I read an article about traffic in {any_city}, it sounds awful.", "City traffic can be really tough."),
    ("What's a quick dinner I can make in 15 minutes?", "A vegetable stir fry with rice is fast and easy."),
    ("Watched a documentary about {any_city}'s old architecture.", "That sounds interesting!"),
    ("My laptop keeps freezing, any tips?", "Try closing background apps and checking for updates."),
    ("How do I make my coffee less bitter?", "Use slightly cooler water and a coarser grind."),
    ("Thinking of starting to run in the mornings.", "Great idea, start slow and build up."),
    ("Any tips for sleeping better?", "Keep a fixed schedule and avoid screens before bed."),
    ("I saw a cricket match highlight from {any_city} today.", "Cricket highlights are always fun to watch."),
    ("{pet} chewed my headphones again.", "Classic pet behavior!"),
    ("What's a good way to learn to cook?", "Start with 5 simple recipes and repeat them."),
]

# Question templates. Each fact has several wordings.
Q_PET = ["What's my pet's name?", "What is the name of my {kind}?", "What did I name my {kind}?"]
Q_SIB_NAME = ["What's my {rel}'s name?", "What is my {rel} called?"]
Q_CITY_NOW = ["Which city do I live in now?", "Where am I based currently?", "What city do I live in?"]
Q_CITY_BEFORE = ["Which city did I live in before I moved?", "Where did I live before?"]
Q_EMAIL = ["What was my old work email?", "What's my work email?"]
Q_SIB_CITY = ["Which city does my {rel} live in?", "Where does my {rel} live?"]


def pick(rng: random.Random, templates: list[tuple[str, str]], **values) -> list[dict]:
    user_msg, assistant_msg = rng.choice(templates)
    return [
        {"role": "user", "content": user_msg.format(**values)},
        {"role": "assistant", "content": assistant_msg.format(**values)},
    ]


def filler_turns(rng: random.Random, n: int, **values) -> list[dict]:
    messages = []
    for template in rng.sample(FILLER, n):
        messages += pick(rng, [template], any_city=rng.choice(CITIES), **values)
    return messages


def make_user(rng: random.Random, index: int, n_sessions: int) -> dict:
    # 1. Choose the facts. Old city, new city and sibling city are all different, so answers cannot be confused.
    old_city, new_city, sib_city = rng.sample(CITIES, 3)
    old_company, new_company = rng.sample(COMPANIES, 2)
    first = rng.choice(FIRST_NAMES)
    rel = rng.choice(["sister", "brother", "cousin"])
    v = {
        "pet": rng.choice(PET_NAMES),
        "kind": rng.choice(PET_KINDS),
        "city": old_city,
        "old_city": old_city,
        "new_city": new_city,
        "sib": rng.choice(SIBLING_NAMES[rel]),
        "rel": rel,
        "sib_city": sib_city,
        "company": old_company,
        "new_company": new_company,
        "email": f"{first.lower()}@{old_company.lower()}.com",
    }

    # 2. Place the fact sessions on the timeline, in order: intro < sibling < move < forget.
    #    Each session is 1-2 filler turns plus the fact turn (if any), so the facts are not always first.
    n_sessions = max(n_sessions, 5)
    slots = sorted(rng.sample(range(1, n_sessions), 3))  # intro is always session 0
    fact_at = {0: "intro", slots[0]: "sibling", slots[1]: "move", slots[2]: "forget"}

    sessions = []
    for i in range(n_sessions):
        messages = filler_turns(rng, rng.randint(1, 2), **v)
        kind = fact_at.get(i)
        if kind == "intro":
            fact = pick(rng, INTRO, **v) + pick(rng, EMAIL, **v)
        elif kind == "sibling":
            fact = pick(rng, SIBLING, **v)
        elif kind == "move":
            fact = pick(rng, MOVE, **v)
        elif kind == "forget":
            fact = pick(rng, FORGET, **v)
        else:
            fact = []
        # Put the fact at a random position inside the session (between filler turns).
        cut = 2 * rng.randint(0, len(messages) // 2)
        messages = messages[:cut] + fact + messages[cut:]
        sessions.append({"session_id": f"s{i + 1}", "messages": messages})

    # 3. Write the questions and the exact answer key.
    def q(templates):
        return rng.choice(templates).format(**v)

    questions = [
        {"question": q(Q_PET), "type": "plain_recall", "answer": v["pet"]},
        {"question": q(Q_SIB_NAME), "type": "plain_recall", "answer": v["sib"]},
        {"question": q(Q_CITY_NOW), "type": "updated_fact", "answer": new_city, "stale_answer": old_city},
        {"question": q(Q_CITY_BEFORE), "type": "previous_value", "answer": old_city},
        {"question": q(Q_EMAIL), "type": "deleted_fact", "answer": "UNKNOWN", "stale_answer": v["email"]},
        {"question": rng.choice(NEVER_MENTIONED), "type": "never_mentioned", "answer": "UNKNOWN"},
        {"question": q(Q_SIB_CITY), "type": "other_person", "answer": sib_city, "distractor": new_city},
    ]

    return {
        "user_id": f"g{index:03d}",
        "facts": v,                      # for debugging only; eval.py does not read it
        "fact_sessions": {k: f"s{i + 1}" for i, k in fact_at.items()},
        "sessions": sessions,
        "questions": questions,
    }


def check(user: dict) -> None:
    """Sanity checks on the answer key. If one fails, the generator has a bug."""
    text = " ".join(m["content"] for s in user["sessions"] for m in s["messages"])
    for q in user["questions"]:
        if q["answer"] != "UNKNOWN":
            assert q["answer"] in text, f"{user['user_id']}: answer {q['answer']!r} is never mentioned"
    facts = user["facts"]
    assert facts["sib"] in SIBLING_NAMES[facts["rel"]], f"{user['user_id']}: {facts['rel']} named {facts['sib']}"
    email = facts["email"]
    forget_session = user["fact_sessions"]["forget"]
    forget_index = int(forget_session[1:]) - 1
    later = " ".join(m["content"] for s in user["sessions"][forget_index:] for m in s["messages"])
    assert email not in later, f"{user['user_id']}: email appears again after the forget request"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--users", type=int, default=10)
    parser.add_argument("--sessions", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="data/generated")
    args = parser.parse_args()

    rng = random.Random(args.seed)  # same seed -> same dataset, so results are reproducible
    os.makedirs(args.out, exist_ok=True)
    for i in range(args.users):
        user = make_user(rng, i, args.sessions)
        check(user)
        with open(os.path.join(args.out, f"{user['user_id']}.json"), "w") as f:
            json.dump(user, f, indent=2)
    print(f"Wrote {args.users} users x {args.sessions} sessions to {args.out}/ (seed={args.seed})")


if __name__ == "__main__":
    main()
