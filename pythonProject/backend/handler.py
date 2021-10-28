import pymysql
from datetime import datetime
sql_db = pymysql.connect(
    user='root',
    passwd='',
    host='127.0.0.1',
    db='test',
    charset='utf8'
)

cursor = sql_db.cursor(pymysql.cursors.DictCursor)


def find_item_one(str):  # str은 프론트에서 받아온 date값
    # dateFormat = "%Y-%m-%d"
    # date = str.strftime(dateFormat)
    date = str

    sql = "SELECT * FROM test.member_enterance where enterance_time = %s"
    cursor.execute(sql, (date))
    result = cursor.fetchall()

    print(result)

time2 = '2021-10-15T19:19:57.769465'
# time1 = datetime(2021, 10, 15)
find_item_one(time2)
