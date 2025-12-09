import os
import datetime

def git_commit_push():
    os.system("git pull")
    os.system("git add .")

    commit_msg = f"Hourly RAG Update - {datetime.datetime.utcnow().isoformat()}"
    os.system(f'git commit -m "{commit_msg}" || echo "No changes to commit"')

    os.system("git push")
