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

def dbConnection():
    sql_db = pymysql.connect(
        user='root',
        passwd='',
        host='127.0.0.1',
        db='test',
        charset='utf8'
    )

    cursor = sql_db.cursor(pymysql.cursors.DictCursor)
    return cursor

def create_table(): #안되면 dbServer에서 직접 테이블 제작, 근데아마 8080 DRF 실행시, 자동으로 setting

    sql = "CREATE TABLE IF NOT EXISTS member_enterance(eid BIGINT, enterance_time DATETIME, memberID BIGINT;"
    sql2 = "ALTER TABLE member_enterance CONSTRAINT eid ADD PRIMARY KEY (ID);"
    sql3 = ""
    cursor.execute(sql)



def insert_item_one(id, time):  #인식된 멤버id, datetime시간

    sql_db = pymysql.connect(
        user='root',
        passwd='',
        host='127.0.0.1',
        db='test',
        charset='utf8'
    )

    cursor = sql_db.cursor(pymysql.cursors.DictCursor)

    dateFormat = "%Y-%m-%d"
    date = time.strftime(dateFormat)
    print(id, date, "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    sql = "INSERT INTO test.member_enterance(memberID, enterance_time) VALUES (%s, %s)"
    cursor.execute(sql, (id, date))
    print("item inserted@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    sql_db.commit()
    sql_db.close()

def find_enterance_one(str):  # str은 프론트에서 받아온 date값
    # dateFormat = "%Y-%m-%d"
    # date = str.strftime(dateFormat)
    date = str

    sql = "SELECT * FROM test.member_enterance where enterance_time = %s"
    cursor.execute(sql, (date))
    result = cursor.fetchone()

    return result

def find_member_enterance(id):
    # eid가 들어옴

    sql = "SELECT * FROM test.member_enterance where eid = %s"
    cursor.execute(sql, (id))
    result = cursor.fetchall()

    return result


def find_enterance_all(time):


    dateFormat = "%Y-%m-%d"
    date = time.strftime(dateFormat)

    sql = "SELECT * FROM test.member_enterance where enterance_time = %s"

    cursor.execute(sql, (date))
    result = cursor.fetchall()

    return result

def find_id_number():
    sql_db = pymysql.connect(
        user='root',
        passwd='',
        host='127.0.0.1',
        db='test',
        charset='utf8'
    )

    cursor = sql_db.cursor(pymysql.cursors.DictCursor)

    sql = "SELECT COUNT(*) FROM test.member_member;"

    cursor.execute(sql)
    result = cursor.fetchone()
    return result['COUNT(*)']

def get_names():
    sql_db = pymysql.connect(
        user='root',
        passwd='',
        host='127.0.0.1',
        db='test',
        charset='utf8'
    )

    cursor = sql_db.cursor(pymysql.cursors.DictCursor)

    sql = "SELECT name FROM test.member_member;"
    cursor.execute(sql)
    result = cursor.fetchall()
    names = []
    for i in result:
        names.append(i["name"])
    return names

### TDD

# time2 = '2021-10-15T19:19:57.769465'
# # time1 = datetime(2021, 10, 15)
# find_item_one(time2)

# pk = str(3)
# time1 = datetime.now()
# insert_item_one(pk, time1)

# table = 'test.member_enterance'
# column = 'enterance_time'
# time1 = datetime(2021, 10, 29)
#
# enterance_all = find_enterance_all(time1)
# print(enterance_all)

# find_id_number()

# get_names()