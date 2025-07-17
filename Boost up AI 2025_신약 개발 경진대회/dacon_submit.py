from dacon_submit_api import dacon_submit_api 
import dotenv
import os
dotenv.load_dotenv()
"""사용법
result = dacon_submit_api.post_submission_file(
file_path='파일경로', 
token='개인 Token', 
cpt_id='대회ID', 
team_name='팀이름', 
memo='submission 메모 내용' )
"""

def dacon_submit(submission_path, memo):
    ####### 수정 필요 #######
    submission_path = submission_path
    token = os.getenv('TOKEN')
    cpt_id = os.getenv('COMPETITION_ID')
    team_name = '하품'
    memo = memo
    #####################

    try:
        result = dacon_submit_api.post_submission_file(
            file_path=submission_path,
            token=token,
            cpt_id=cpt_id,
            team_name=team_name,
            memo=memo
        )
        print(f"제출 성공: {result}")
    except Exception as e:
        print(f"제출 오류: {e}")