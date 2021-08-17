# 고객 정보 테이블
DROP TABLE IF EXISTS customer;

CREATE TABLE customer(
	id VARCHAR(10),        # id 10자리
	pwd VARCHAR(15),       # pwd 15자리
	name VARCHAR(10),
	age INT,
	height FLOAT,
	weight INT
);

ALTER TABLE customer ADD PRIMARY KEY(id);
ALTER TABLE customer MODIFY COLUMN pwd VARCHAR(15) NOT NULL;
ALTER TABLE customer MODIFY COLUMN name VARCHAR(10) NOT NULL;
ALTER TABLE customer MODIFY COLUMN age INT NOT NULL;
ALTER TABLE customer MODIFY COLUMN height FLOAT NOT NULL;
ALTER TABLE customer MODIFY COLUMN weight INT NOT NULL;

INSERT INTO customer VALUES('id01', 'pwd01', '김영희', 31, 183.1, 63);


# 웹사이트 링크 테이블
DROP TABLE IF EXISTS link;

CREATE TABLE link(
	id INT,
	size CHAR(5),
	mf VARCHAR(500),
	yoox VARCHAR(500)
);

ALTER TABLE link ADD PRIMARY KEY(id);
ALTER TABLE link MODIFY COLUMN id INT AUTO_INCREMENT;
ALTER TABLE link AUTO_INCREMENT=10;
ALTER TABLE link MODIFY COLUMN size CHAR(5) NOT NULL;
ALTER TABLE link MODIFY COLUMN mf VARCHAR(500) NOT NULL;
ALTER TABLE link MODIFY COLUMN yoox VARCHAR(500) NOT NULL;

INSERT INTO link VALUES(NULL, 'XXS', 'https://www.matchesfashion.com/kr/womens/shop?q=%3A%3AclothesSize%3A10001', 'https://www.yoox.com/kr/%EC%97%AC%EC%84%B1/%EC%9D%98%EB%A5%98/shoponline#/dept=clothingwomen&gender=D&page=1&size=1&season=X');
INSERT INTO link VALUES(NULL, 'S', 'https://www.matchesfashion.com/kr/womens/shop?q=%3A%3AclothesSize%3A10003', 'https://www.yoox.com/kr/%EC%97%AC%EC%84%B1/%EC%9D%98%EB%A5%98/shoponline#/dept=clothingwomen&gender=D&page=1&size=3&season=X');
INSERT INTO link VALUES(NULL, 'M', 'https://www.matchesfashion.com/kr/womens/shop?q=%3A%3AclothesSize%3A10004', 'https://www.yoox.com/kr/%EC%97%AC%EC%84%B1/%EC%9D%98%EB%A5%98/shoponline#/dept=clothingwomen&gender=D&page=1&size=4&season=X');
INSERT INTO link VALUES(NULL, 'L', 'https://www.matchesfashion.com/kr/womens/shop?q=%3A%3AclothesSize%3A10005', 'https://www.yoox.com/kr/%EC%97%AC%EC%84%B1/%EC%9D%98%EB%A5%98/shoponline#/dept=clothingwomen&gender=D&page=1&size=5&season=X');
INSERT INTO link VALUES(NULL, 'XL', 'https://www.matchesfashion.com/kr/womens/shop?q=%3A%3AclothesSize%3A10006', 'https://www.yoox.com/kr/%EC%97%AC%EC%84%B1/%EC%9D%98%EB%A5%98/shoponline#/dept=clothingwomen&gender=D&page=1&size=6&season=X');
INSERT INTO link VALUES(NULL, 'XXL', 'https://www.matchesfashion.com/kr/womens/shop?q=%3A%3AclothesSize%3A10007', 'https://www.yoox.com/kr/%EC%97%AC%EC%84%B1/%EC%9D%98%EB%A5%98/shoponline#/dept=clothingwomen&gender=D&page=1&size=7&season=X');
INSERT INTO link VALUES(NULL, 'XXXL', 'https://www.matchesfashion.com/kr/womens/shop?q=%3A%3AclothesSize%3A10272', 'https://www.yoox.com/kr/%EC%97%AC%EC%84%B1/%EC%9D%98%EB%A5%98/shoponline#/dept=clothingwomen&gender=D&page=1&size=8&season=X');
