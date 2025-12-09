import requests

def fetch_stackoverflow_qa():
    url = "https://api.stackexchange.com/2.3/questions"
    params = {
        "order": "desc",
        "sort": "activity",
        "site": "stackoverflow",
        "pagesize": 10,
        "filter": "withbody"
    }

    q_data = requests.get(url, params=params).json()["items"]

    qa_pairs = []
    for q in q_data:
        answer_url = f"https://api.stackexchange.com/2.3/questions/{q['question_id']}/answers"
        ans = requests.get(answer_url, params={"site": "stackoverflow"}).json()

        if ans["items"]:
            qa_pairs.append({
                "question": q["title"],
                "answer": ans["items"][0].get("body", ""),
                "tags": q["tags"]
            })

    return qa_pairs
