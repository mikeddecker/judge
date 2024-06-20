CREATE DATABASE IF NOT EXISTS judge;

CREATE TABLE Folders (
    FolderID INT PRIMARY KEY AUTO_INCREMENT,
    Path NVARCHAR(50) NOT NULL
);

CREATE TABLE Videos (
    VideoID INT PRIMARY KEY AUTO_INCREMENT,
    FolderID INT NOT NULL,
    Name NVARCHAR(255) NOT NULL,
    Training BOOLEAN NOT NULL,
    Obstruction TINYINT NOT NULL, -- 0 geen - 100 high obstruction
    Quality TINYINT NOT NULL, -- 0 zeer wazig - 100 goed
    DimensionX SMALLINT NOT NULL, -- 1920 of 1080
    DimensionY SMALLINT NOT NULL, -- 1080 of 1920 of 3k...

    FOREIGN KEY (FolderID) REFERENCES Folders(FolderID)
);

CREATE TABLE FrameLabels (
    VideoID INT NOT NULL,
    FrameNr SMALLINT NOT NULL,
    Label TINYINT NOT NULL, -- 0 - 9 currently (on ground, air, release, power...)
    Verrified BOOLEAN NOT NULL, -- 0 not, 1
    
    FOREIGN KEY (VideoID) REFERENCES Videos(VideoID)
);
