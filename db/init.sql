CREATE DATABASE IF NOT EXISTS judge;

DROP TABLE IF EXISTS FrameLabels;
DROP TABLE IF EXISTS BorderLabels;
DROP TABLE IF EXISTS Borders;
DROP TABLE IF EXISTS Videos;
DROP TABLE IF EXISTS Folders;

CREATE TABLE Folders (
    folderID INT PRIMARY KEY NOT NULL AUTO_INCREMENT,
    path NVARCHAR(50) NOT NULL
);

CREATE TABLE Videos (
    videoID INT PRIMARY KEY NOT NULL AUTO_INCREMENT,
    folderID INT NOT NULL,
    name NVARCHAR(255) NOT NULL,
    training BOOLEAN NOT NULL,
    obstruction TINYINT NOT NULL, -- 0 geen - 100 high obstruction
    quality TINYINT NOT NULL, -- 0 zeer wazig - 100 goed
    type TINYINT NOT NULL, -- 0 = competitie, 1 = distant, sporthal-like, 2 = dicht & vrij, 3 = distant, sporthallike, ruis, 4 = dicht & achtergrondruis
    dimensionX SMALLINT NOT NULL, -- 1920 of 1080
    dimensionY SMALLINT NOT NULL, -- 1080 of 1920 of 3k...

    FOREIGN KEY (folderID) REFERENCES Folders(folderID)
);

CREATE TABLE FrameLabels ( -- CNN based
    videoID INT NOT NULL,
    frameNr SMALLINT NOT NULL,
    label TINYINT NOT NULL, -- 0 - 9 currently (on ground, air, release, power...)
    manual_insert BOOLEAN NOT NULL, -- 0 not, 1 manual
    
    FOREIGN KEY (videoID) REFERENCES Videos(videoID)
);

CREATE TABLE BorderLabels ( -- Interval based
    videoID INT NOT NULL,
    frameNr SMALLINT NOT NULL,
    label TINYINT NOT NULL, -- 0 no skill, 1 start, 2 executing skill, 3 skill end
    manual_insert BOOLEAN NOT NULL, -- 0 not, 1 manual
    
    FOREIGN KEY (videoID) REFERENCES Videos(videoID)
);

CREATE TABLE Borders (
    videoID INT NOT NULL,
    frame_start SMALLINT NOT NULL,
    frame_end SMALLINT NOT NULL,
    manual_insert BOOLEAN NOT NULL, -- 0 not, 1 manual
    
    FOREIGN KEY (videoID) REFERENCES Videos(videoID)
)