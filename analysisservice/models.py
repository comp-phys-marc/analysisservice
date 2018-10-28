from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import relationship
from .database import Base


class User(Base):

    __tablename__ = 'User'

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255))
    email = Column(String(255), unique=True)
    password = Column(String(255))
    companyName = Column(String(255))
    phone = Column(String(50))
    calculations = relationship("Calculation", primaryjoin=Calculation.user_id == id)

    def __init__(self, name, email, phone, password, companyName):
        self.name = name
        self.email = email
        self.phone = phone
        self.password = password
        self.companyName = companyName

    def __repr__(self):
        return '<User %r>' % (self.name)


class Calculation(Base):

    __tablename__ = 'Calculation'

    id = Column(Integer, primary_key=True, autoincrement=True)
    result = Column(String(2048))
    user_id = Column(Integer, ForeignKey('User.id'))
    type = Column(String(255))
    phases = relationship("SinglePhase", primaryjoin=SinglePhase.calculation_id == id)

    def __init__(self, result, user_id, type):
        self.result = result
        self.user_id = user_id
        self.type = type

    def __repr__(self):
        return '<Calculation %r>' % (self.id)


class SinglePhase(Base):

    __tablename__ = 'Phase'

    id = Column(Integer, primary_key=True, autoincrement=True)
    imaginary_flags = Column(String(2048))
    negative_flags = Column(String(2048))
    states = Column(String(2048))
    previous_phase_id = Column(Integer)
    phase = Column(Integer)
    calculation_id = Column(Integer, ForeignKey('Calculation.id'))

    def __init__(self, previous_phase_id, phase, imaginary_flags, negative_flags, states, calculation_id):
        self.previous_phase_id = previous_phase_id
        self.phase = phase
        self.imaginary_flags = imaginary_flags
        self.negative_flags = negative_flags
        self.states = states
        self.calculation_id = calculation_id

    def __repr__(self):
        return '<SinglePhase %r>' % (self.id)

