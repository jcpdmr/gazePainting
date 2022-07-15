from sqlalchemy import create_engine, Column, DateTime, String, Float, Integer
from sqlalchemy.ext.declarative import declarative_base

# In case of SQLAlchemy Debug put echo=True
engine = create_engine("sqlite:///database/history.sqlite3", echo=False)
base = declarative_base()


class history(base):
    __tablename__ = "history"
    datetime = Column(DateTime, primary_key=True)
    painting_name = Column(String)
    seconds = Column(Float)
    gender = Column(String)
    gender_score = Column(Integer)
    age = Column(String)
    age_score = Column(Integer)

    def __init__(self, datetime, painting_name, seconds, gender, gender_score, age, age_score):
        self.datetime = datetime
        self.painting_name = painting_name
        self.seconds = seconds
        self.gender = gender
        self.gender_score = gender_score
        self.age = age
        self.age_score = age_score


base.metadata.create_all(engine)
