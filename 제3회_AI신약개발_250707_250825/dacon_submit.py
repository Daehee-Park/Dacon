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

####### 수정 필요 #######
file_path = 'output/baseline_cv_submission.csv'
token = os.getenv('TOKEN')
cpt_id = os.getenv('COMPETITION_ID')
team_name = '하품'
memo = 'MEMO'
#####################

result = dacon_submit_api.post_submission_file(
    file_path=file_path,
    token=token,
    cpt_id=cpt_id,
    team_name=team_name,
    memo=memo
)
print(f"제출 결과: {result}")