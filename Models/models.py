from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import ForeignKey


db = SQLAlchemy()

class Artist(db.Model):
    __tablename__ = 'artists'
    ArtistId: Mapped[int] = mapped_column(primary_key=True)
    Name: Mapped[str] = mapped_column(db.String(120), nullable=False)

    albums = db.relationship('Album', back_populates='artist')

    def to_dict(self):
        return {'ArtistId': self.ArtistId, 'Name': self.Name}

class Album(db.Model):
    __tablename__ = 'albums'
    AlbumId: Mapped[int] = mapped_column(primary_key=True)
    Title: Mapped[str] = mapped_column(db.String(160), nullable=False)
    ArtistId: Mapped[int] = mapped_column(ForeignKey('artists.ArtistId'))

    
    artist = db.relationship('Artist', back_populates='albums')

    def to_dict(self):
        return {'AlbumId': self.AlbumId, 'Title': self.Title, 'ArtistId': self.ArtistId}
    

    