from __future__ import annotations
from sqlalchemy.orm import declarative_base, relationship, sessionmaker, column_property
from sqlalchemy import (
    Column,
    Integer,
    String,
    create_engine,
    Table,
    Text,
    ForeignKey,
    Float,
)
from pydantic import BaseModel
from typing import List, Dict, Set
from enum import Enum
import numpy as np
import json
from tqdm import tqdm


class HoldRole(str, Enum):
    start = "start"
    middle = "middle"
    finish = "finish"
    foot = "foot"
    # we should never see these values with our board layout
    cyan = "cyan"
    magenta = "magenta"
    yellow = "yellow"
    green = "green"
    red = "red"
    blue = "blus"

HOLD_ROLE_ID = {HoldRole.start: 1, HoldRole.middle: 2, HoldRole.finish: 3, HoldRole.foot: 4}

class HoldData(BaseModel, frozen=True):
    role: HoldRole
    # hole data (hold != hole)
    hole_id: int
    name: str  # related to position ex: (12,36)
    x: int  # actually starts at 4 (name.x * 4)
    y: int  # name.y * 4 + 8

    def relative_xy_and_distance(self, other: HoldData):
        # x,y plan is the board. So x,y are not influenced by the angle.
        # We ignore the fact that the first two rows do not move with the angle.
        rel_x = other.x - self.x
        rel_y = other.y - self.y
        distance = np.sqrt(rel_x**2 + rel_y**2)
        return rel_x, rel_y, distance


class ClimbData(BaseModel):
    uuid: str
    name: str
    angle: str
    holds: List[HoldData]
    difficulty_average: float
    quality_average: float
    ascensionist_count: int

    def img(self):
        MAX_X = 140 // 4
        MAX_Y = 152 // 4
        img = np.zeros((MAX_Y, MAX_X, 2), dtype=np.uint8)
        img[..., 1] = int(self.angle)
        
        # replace with fast np.where or other
        for hold in self.holds:
            col = (hold.x // 4) - 1
            row = (MAX_Y - hold.y //4) -1
            img[row, col, 0] = HOLD_ROLE_ID[hold.role]

        return img



Base = declarative_base()

class Layout(Base):
    __tablename__ = "layouts"
    id = Column(Integer, primary_key=True)
    product_id = Column(Integer)

class Climb(Base):
    __tablename__ = "climbs"
    uuid = Column(Text, primary_key=True)
    layout_id = Column(Integer, ForeignKey(Layout.id))
    layout_deviant_id = Column(Integer)
    name = Column(Text)
    stats = relationship("ClimbStats", uselist=False)

    # utils
    climb_placements = relationship("ClimbPlacement")

    def __repr__(self):
        return f"{self.uuid} {self.name}"

    def get_climb_data(self) -> ClimbData:
        return ClimbData(
            uuid=self.uuid,
            name=self.name,
            angle=self.stats.angle,
            holds=[cp.get_hold_data() for cp in self.climb_placements],
            difficulty_average=self.stats.difficulty_average,
            quality_average=self.stats.quality_average,
            ascensionist_count=self.stats.ascensionist_count,
        )


class ClimbStats(Base):
    __tablename__ = "climb_stats"
    climb_uuid = Column(Text, ForeignKey(Climb.uuid), primary_key=True)
    angle = Column(Integer)
    difficulty_average = Column(Float)
    quality_average = Column(Float)
    ascensionist_count = Column(Integer)


class Hole(Base):
    __tablename__ = "holes"
    id = Column(Integer, primary_key=True)
    product_id = Column(Integer)
    name = Column(Text)
    x = Column(Integer)
    y = Column(Integer)


class Placement(Base):
    __tablename__ = "placements"
    id = Column(Integer, primary_key=True)
    layout_id = Column(Integer)
    hole_id = Column(Integer, ForeignKey(Hole.id))
    hold_id = Column(Integer)
    default_placement_role_id = Column(Integer)

    hole = relationship("Hole")


class PlacementRole(Base):
    __tablename__ = "placement_roles"
    id = Column(Integer, primary_key=True)
    product_id = Column(Integer)
    position = Column(Integer)
    name = Column(Text)
    full_name = Column(Text)
    led_color = Column(Text)
    screen_color = Column(Text)


class ClimbPlacement(Base):
    __tablename__ = "climbs_placements"
    climb_uuid = Column(Text, ForeignKey(Climb.uuid), primary_key=True)
    placement_id = Column(Integer, ForeignKey(Placement.id), primary_key=True)
    frames = Column(Text)
    role_id = Column(Integer, ForeignKey(PlacementRole.id))

    # utils
    role = relationship("PlacementRole", uselist=False)
    placement = relationship("Placement", uselist=False)
    climb = relationship("Climb", uselist=False)

    def get_hold_data(self):
        return HoldData(
            role=self.role.name,
            hole_id=self.placement.hole_id,
            name=self.placement.hole.name,
            x=self.placement.hole.x,
            y=self.placement.hole.y,
        )

def all_climbs():
    engine = create_engine(
        "sqlite:///file:climbing/db.sqlite3?mode=ro&nolock=1&uri=true",
        # echo=True
    )
    Session = sessionmaker()
    Session.configure(bind=engine)
    s = Session()
    climbs = s.query(Climb).filter(Climb.stats.has()).filter(Climb.layout_id == 1, Climb.layout_deviant_id == 9)
    return climbs

def main():
    climbs = all_climbs()
    # climbs_json = json.dumps([climb.get_climb_data().dict() for climb in tqdm(climbs.limit(100))])
    climb = list(climbs.limit(1))[0]
    climb_data = climb.get_climb_data()
    img = climb_data.img() * 50

    import cv2
    cv2.imwrite("test.png", img[..., 0])

    print("done")

if __name__ == "__main__":
    main()
