USE judge;

CREATE OR REPLACE VIEW SkillTrainDistribution AS
SELECT sds.name, COUNT(sd.skill) as skill_count,
COUNT(sd.skill) * 100 /c.total as total
FROM Skillinfo_DoubleDutch sd
JOIN Skillinfo_DoubleDutch_Skill sds ON sd.skill = sds.id
JOIN (SELECT COUNT(*) as total FROM Skillinfo_DoubleDutch WHERE MOD(videoId, 10) <> 5) c
WHERE MOD(videoId, 10) <> 5
GROUP BY sd.skill, sds.name, c.total
ORDER BY skill_count DESC;

CREATE OR REPLACE VIEW SkillValDistribution AS
SELECT sds.name, COUNT(sd.skill) as skill_count,
COUNT(sd.skill) * 100 /c.total as total
FROM Skillinfo_DoubleDutch sd
JOIN Skillinfo_DoubleDutch_Skill sds ON sd.skill = sds.id
JOIN (SELECT COUNT(*) as total FROM Skillinfo_DoubleDutch WHERE MOD(videoId, 10) = 5) c
WHERE MOD(videoId, 10) = 5
GROUP BY sd.skill, sds.name, c.total
ORDER BY skill_count DESC;

CREATE OR REPLACE VIEW SkillTrainTestComparison_Skills AS
SELECT t.total as train_percentage, t.skill_count as train_count, t.name as skillname, v.skill_count as val_count, v.total as val_total
FROM SkillTrainDistribution t
LEFT JOIN SkillValDistribution v
ON t.name = v.name;

/* Skill distribution turners */ 

CREATE OR REPLACE VIEW SkillTrainDistribution_Turner1 AS
SELECT sdt.name, COUNT(sd.turner1) as turner1_count,
COUNT(sd.turner1) * 100 /c.total as total
FROM Skillinfo_DoubleDutch sd
JOIN Skillinfo_DoubleDutch_Turner sdt ON sd.turner1 = sdt.id
JOIN (SELECT COUNT(*) as total FROM Skillinfo_DoubleDutch WHERE MOD(videoId, 10) <> 5) c
WHERE MOD(videoId, 10) <> 5
GROUP BY sd.turner1, sdt.name, c.total
ORDER BY turner1_count DESC;

CREATE OR REPLACE VIEW SkillValDistribution_Turner1 AS
SELECT sdt.name, COUNT(sd.turner1) as turner1_count,
COUNT(sd.turner1) * 100 /c.total as total
FROM Skillinfo_DoubleDutch sd
JOIN Skillinfo_DoubleDutch_Turner sdt ON sd.turner1 = sdt.id
JOIN (SELECT COUNT(*) as total FROM Skillinfo_DoubleDutch WHERE MOD(videoId, 10) = 5) c
WHERE MOD(videoId, 10) = 5
GROUP BY sd.turner1, sdt.name, c.total
ORDER BY turner1_count DESC;

CREATE OR REPLACE VIEW SkillTrainTestComparison_Turner1 AS
SELECT t.total as train_percentage, t.turner1_count as train_count, t.name as turnername, v.turner1_count as val_percentage, v.total as val_total
FROM Skillinfo_DoubleDutch_Turner sdt
LEFT JOIN SkillTrainDistribution_Turner1 t ON sdt.name = t.name
LEFT JOIN SkillValDistribution_Turner1 v ON sdt.name = v.name;

CREATE OR REPLACE VIEW SkillTrainDistribution_Turner2 AS
SELECT sdt.name, COUNT(sd.turner2) as turner2_count,
COUNT(sd.turner2) * 100 /c.total as total
FROM Skillinfo_DoubleDutch sd
JOIN Skillinfo_DoubleDutch_Turner sdt ON sd.turner2 = sdt.id
JOIN (SELECT COUNT(*) as total FROM Skillinfo_DoubleDutch WHERE MOD(videoId, 10) <> 5) c
WHERE MOD(videoId, 10) <> 5
GROUP BY sd.turner2, sdt.name, c.total
ORDER BY turner2_count DESC;

CREATE OR REPLACE VIEW SkillValDistribution_Turner2 AS
SELECT sdt.name, COUNT(sd.turner2) as turner2_count,
COUNT(sd.turner2) * 100 /c.total as total
FROM Skillinfo_DoubleDutch sd
JOIN Skillinfo_DoubleDutch_Turner sdt ON sd.turner2 = sdt.id
JOIN (SELECT COUNT(*) as total FROM Skillinfo_DoubleDutch WHERE MOD(videoId, 10) = 5) c
WHERE MOD(videoId, 10) = 5
GROUP BY sd.turner2, sdt.name, c.total
ORDER BY turner2_count DESC;

CREATE OR REPLACE VIEW SkillTrainTestComparison_Turner2 AS
SELECT t.total as train_percentage, t.turner2_count as train_count, t.name as turnername, v.turner2_count as val_percentage, v.total as val_total
FROM Skillinfo_DoubleDutch_Turner sdt
LEFT JOIN SkillTrainDistribution_Turner2 t ON sdt.name = t.name
LEFT JOIN SkillValDistribution_Turner2 v ON sdt.name = v.name;

/* Skill distribution turners end */


CREATE OR REPLACE VIEW Skills_labeled_per_day AS
SELECT so.labeldate, ji.count as jumps_included, so.count as skills
FROM (SELECT labeldate, COUNT(*) as count FROM Skillinfo_DoubleDutch
GROUP BY labeldate) AS ji -- jumps_included
JOIN (SELECT labeldate, COUNT(*) as count FROM Skillinfo_DoubleDutch WHERE skill <> 1
GROUP BY labeldate) AS so -- skills only 
ON ji.labeldate = so.labeldate;

CREATE OR REPLACE VIEW SkillTrainTestComparison_Skills_5classes AS
SELECT 
    CASE 
        WHEN train_count >= 400 THEN skillname
        ELSE 'other'
    END AS skillname,
    SUM(train_count) AS train_count,
    SUM(train_percentage) AS train_percentage,
    SUM(val_count) AS val_count,
    SUM(val_total) AS val_total
FROM SkillTrainTestComparison_Skills
GROUP BY 
    CASE 
        WHEN train_count >= 400 THEN skillname
        ELSE 'other'
    END
ORDER BY train_count DESC;

CREATE OR REPLACE VIEW SkillLengthCountInFrames AS
SELECT length, COUNT(length) 
FROM (SELECT frameEnd - frameStart as length FROM Skillinfo_DoubleDutch) AS skilllengths
GROUP BY length
ORDER BY length;