# Link table 접속
from frame.db import Db
from frame.sql import Sql


class LinkDB(Db):
    def selectOne(self, size):
        conn = super().getConn()
        cursor = conn.cursor()
        cursor.execute(Sql.linkselone %(size))
        links = cursor.fetchone()    # link tuple
        super().close(cursor, conn)
        return links



if __name__ == '__main__':
    print(LinkDB().selectOne('M'))