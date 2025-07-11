SET FOREIGN_KEY_CHECKS = 0; -- disable a foreign keys check
SET AUTOCOMMIT = 0; -- disable autocommit
START TRANSACTION; -- begin transaction

TRUNCATE Table Videos;
ALTER TABLE Videos AUTO_INCREMENT = 1;
TRUNCATE Table CompetitionInfo;
ALTER TABLE CompetitionInfo AUTO_INCREMENT = 1;
TRUNCATE Table Folders;
ALTER TABLE Folders AUTO_INCREMENT = 1;
TRUNCATE Table FrameLabels;
ALTER TABLE FrameLabels AUTO_INCREMENT = 1;
TRUNCATE Table FrameLabelTypes;
ALTER TABLE FrameLabelTypes AUTO_INCREMENT = 1;
TRUNCATE Table Jobs;
ALTER TABLE Jobs AUTO_INCREMENT = 1;
TRUNCATE Table ML_Model;
ALTER TABLE ML_Model AUTO_INCREMENT = 1;
TRUNCATE Table Sources;
ALTER TABLE Sources AUTO_INCREMENT = 1;
TRUNCATE Table TrainResults;
ALTER TABLE TrainResults AUTO_INCREMENT = 1;
TRUNCATE Table TrainResultsSkills;
ALTER TABLE TrainResultsSkills AUTO_INCREMENT = 1;

SET FOREIGN_KEY_CHECKS = 1; -- enable a foreign keys check
COMMIT;  -- make a commit
SET AUTOCOMMIT = 1 ;