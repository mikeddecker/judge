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

CREATE TABLE CompetitionType (
    competitionID INT PRIMARY KEY NOT NULL AUTO_INCREMENT,
    competitionName CHAR(255) NOT NULL
);

CREATE TABLE Club (
    clubID INT PRIMARY KEY NOT NULL AUTO_INCREMENT,
    clubName CHAR(255) NOT NULL
);

CREATE TABLE Discipline (
    disciplineID INT PRIMARY KEY NOT NULL AUTO_INCREMENT,
    disciplineName CHAR(255) NOT NULL
);

CREATE TABLE Age (
    ageID INT PRIMARY KEY NOT NULL AUTO_INCREMENT,
    ageCategory CHAR(255) NOT NULL
);

CREATE TABLE Videos (
    videoID INT PRIMARY KEY NOT NULL AUTO_INCREMENT,
    folderID INT NOT NULL,
    name NVARCHAR(255) NOT NULL,
    training BOOLEAN NULL DEFAULT NULL,
    obstruction TINYINT NOT NULL, -- 0 geen - 100 high obstruction
    manually_bordered TINYINT NOT NULL DEFAULT 0,
    borderlabels_added TINYINT NOT NULL DEFAULT 0,
    competitionID INT NOT NULL, -- PK, BK, WK, EK, Olympics, no competition
    clubID INT NOT NULL, -- sipiro, moving, unknown...
    disciplineID INT NOT NULL, -- SR, CW, SR2, DD3...
    ageID INT NOT NULL, -- senioren, junioren, junsen (= junioren of senioren)
    vid_frame_length INT NULL,

    FOREIGN KEY (folderID) REFERENCES Folders(folderID),
    FOREIGN KEY (competitionID) REFERENCES CompetitionType(competitionID),
    FOREIGN KEY (clubID) REFERENCES Club(clubID),
    FOREIGN KEY (ageID) REFERENCES Age(ageID)
);

CREATE TABLE FrameLabels ( -- CNN based
    videoID INT NOT NULL,
    frameNr SMALLINT NOT NULL, -- max 32k
    label TINYINT NULL, -- 0 no skill, 1 skillborder, 2 skill
    manual_insert BOOLEAN NOT NULL, -- 0 not, 1 manual
    rect_center_x FLOAT(3) DEFAULT NULL,
    rect_center_y FLOAT(3) DEFAULT NULL,
    rect_size FLOAT(3) DEFAULT NULL,
    
    FOREIGN KEY (videoID) REFERENCES Videos(videoID)
);

CREATE TABLE `Borders` (
    videoID int NOT NULL,
    frame_start smallint NOT NULL,
    frame_end smallint NOT NULL,
    manual_insert tinyint(1) NOT NULL,
    skip_in_learning BOOLEAN NOT NULL DEFAULT 0, -- whether or not to skipp a part when learning.

    FOREIGN KEY (videoID) REFERENCES Videos(videoID)
)


DELIMITER $$

/**
    Procedure: Frame batch_nrs order
    e.g.
    1) video_id 2, batch 5
    2) video_id 8, batch 66
    3) video_id 3, batch 27
    4) video_id 1, batch 21
    5) video_id 2, batch 66
    ...
*/

DROP PROCEDURE IF EXISTS GetFrameBatchNrs$$
CREATE PROCEDURE GetFrameBatchNrs (IN batch_size INT, IN train BOOLEAN)
BEGIN
	WITH train_or_val_idxs AS (
		SELECT videoID FROM Videos WHERE training = train
    ), frame_batch_nrs AS (
		SELECT *, FLOOR((frameNr-1) / 16) as batch_nr_video FROM FrameLabels 
        WHERE videoID IN (SELECT * FROM train_or_val_idxs) AND rect_center_x IS NOT NULL
	)
	SELECT videoID, batch_nr_video, ROW_NUMBER() OVER (ORDER BY RAND()) - 1 AS batch_id
	FROM frame_batch_nrs 
    GROUP by videoID, batch_nr_video
    ORDER BY batch_id;
END$$


/**
    Procedure: getting the i'th batch
    VideoID = 2, batch_nr = 3, batch_size = 16
    --> frameLabels list from [33, 48] 
*/
DROP PROCEDURE IF EXISTS GetRectLabels$$
CREATE PROCEDURE GetRectLabels(video_id INT, batch_nr INT, batch_size INT)
BEGIN
	SELECT * FROM FrameLabels 
    WHERE videoID = video_id AND FrameNr BETWEEN batch_nr * batch_size + 1 AND (batch_nr + 1) * batch_size AND rect_center_x IS NOT NULL
    ORDER BY FrameNr;
END$$

DELIMITER ;