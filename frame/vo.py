# Value Object

class CustVO:
    def __init__(self, id, pwd, name, age, height, weight):
        self.__id = id
        self.__pwd = pwd
        self.__name = name
        self.__age = age
        self.__weight = weight
        self.__height = height

    def getId(self):
        return self.__id

    def getPwd(self):
        return self.__pwd

    def getName(self):
        return self.__name

    def getAge(self):
        return self.__age

    def getHt(self):
        return self.__height

    def getWt(self):
        return self.__weight

    def __str__(self):
        return self.__id + ' ' + self.__pwd + ' ' + self.__name + ' ' + str(self.__age) + \
               ' ' + str(self.__height) + ' ' + str(self.__weight)



if __name__ == '__main__':
    cust = CustVO('id01', 'pwd', '김영희', 31, 183.5, 63)
    print(cust)

