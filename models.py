from sqlalchemy import Column, Integer, String, Text, ForeignKey, DateTime
from datetime import datetime
from database import Base


class ChatSession(Base):
    __tablename__ = "Sessions"
    id = Column(Integer, primary_key = True, index = True)
    namespace = Column(String, unique = True, index = True)
    url = Column(String)
    created_at = Column(DateTime, default = datetime.utcnow)

class Message(Base):
    __tablename__ = "Messages"
    id = Column(Integer, primary_key = True, index = True)
    session_id = Column(Integer, ForeignKey("Sessions.id"))
    role = Column(String)
    message = Column(Text)
    timestamp = Column(DateTime, default = datetime.utcnow)
        