class Sql:
    custselone = "SELECT * FROM customer WHERE id='%s'"
    custselall = "SELECT * FROM customer"
    custinsert = "INSERT INTO customer VALUES ('%s', '%s', '%s', %d, %f, %d)"
    custupdate = "UPDATE customer SET pwd='%s', name='%s', age=%d, height=%f, weight=%d WHERE id='%s'"
    custdelete = "DELETE FROM customer WHERE id='%s'"

    linkselone = "SELECT mf, yoox FROM link WHERE size='%s'"