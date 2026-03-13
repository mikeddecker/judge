DELIMITER $$

CREATE PROCEDURE AddCalculatedColumnIfNotExists(
    IN tableName VARCHAR(64),
    IN columnName VARCHAR(64),
    IN columnType VARCHAR(64),
    IN expression TEXT
)
BEGIN
    -- Check if the column already exists
    IF NOT EXISTS (
        SELECT 1 
        FROM INFORMATION_SCHEMA.COLUMNS 
        WHERE TABLE_NAME = tableName AND COLUMN_NAME = columnName
    ) THEN
        -- Construct and execute the ALTER TABLE statement
        SET @sql = CONCAT(
            'ALTER TABLE ', tableName, 
            ' ADD COLUMN ', columnName, ' ', columnType,
            ' AS (', expression, ') STORED'
        );
        PREPARE stmt FROM @sql;
        EXECUTE stmt;
        DEALLOCATE PREPARE stmt;
    END IF;
END$$

DELIMITER ;

CALL AddCalculatedColumnIfNotExists('Accounts', 'uuid_str', 'VARCHAR(36)', 'BIN_TO_UUID(id)');
CALL AddCalculatedColumnIfNotExists('FrameLabels', 'uuid_str', 'VARCHAR(36)', 'BIN_TO_UUID(id)');
CALL AddCalculatedColumnIfNotExists('Folders', 'uuid_str', 'VARCHAR(36)', 'BIN_TO_UUID(id)');
CALL AddCalculatedColumnIfNotExists('Jobs', 'uuid_str', 'VARCHAR(36)', 'BIN_TO_UUID(id)');
CALL AddCalculatedColumnIfNotExists('Layers', 'uuid_str', 'VARCHAR(36)', 'BIN_TO_UUID(id)');
CALL AddCalculatedColumnIfNotExists('LayerValues', 'uuid_str', 'VARCHAR(36)', 'BIN_TO_UUID(id)');
CALL AddCalculatedColumnIfNotExists('LayerComposition', 'uuid_str', 'VARCHAR(36)', 'BIN_TO_UUID(id)');
CALL AddCalculatedColumnIfNotExists('Skills', 'uuid_str', 'VARCHAR(36)', 'BIN_TO_UUID(id)');
CALL AddCalculatedColumnIfNotExists('Tags', 'uuid_str', 'VARCHAR(36)', 'BIN_TO_UUID(id)');
CALL AddCalculatedColumnIfNotExists('TagGroups', 'uuid_str', 'VARCHAR(36)', 'BIN_TO_UUID(id)');
CALL AddCalculatedColumnIfNotExists('TrainResults', 'uuid_str', 'VARCHAR(36)', 'BIN_TO_UUID(id)');
CALL AddCalculatedColumnIfNotExists('TrainResultsEpoch', 'uuid_str', 'VARCHAR(36)', 'BIN_TO_UUID(id)');
CALL AddCalculatedColumnIfNotExists('Videos', 'uuid_str', 'VARCHAR(36)', 'BIN_TO_UUID(id)');

