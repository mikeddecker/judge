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
SELECT t.total as train_percentage, t.skill_count as train_count, t.name as skillname, v.skill_count as val_percentage, v.total as val_total
FROM SkillTrainDistribution t
JOIN SkillValDistribution v
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
