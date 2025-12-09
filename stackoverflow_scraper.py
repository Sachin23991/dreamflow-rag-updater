import requests

def fetch_stackoverflow_qa():
    q_url = "https://api.stackexchange.com/2.3/questions"
    params = {
        "order": "desc",
        "sort": "activity",
        "site": "stackoverflow",
        "pagesize": 10,
        "filter": "withbody"
    }

    questions = requests.get(q_url, params=params).json()["items"]
    qa_pairs = []

    for q in questions:
        ans_url = f"https://api.stackexchange.com/2.3/questions/{q['question_id']}/answers"
        ans_data = requests.get(ans_url, params={"site": "stackoverflow", "filter": "withbody"}).json()

        if ans_data["items"]:
            qa_pairs.append({
                "question": q["title"],
                "answer": ans_data["items"][0]["body"],
                "tags": q["tags"]
            })

    return qa_pairs
