# Disaster Recovery & Operational Runbooks

**Last Updated**: March 14, 2026
**See also**: [[AI-Judge.Roadmap]] (Phase 3), [[AI-Judge.Multi-Region]], [[AI-Judge.Backup]], [[AI-Judge.Availability]]

---

## Recovery Objectives

| Metric                             | Target              | Current Status               |
| ---------------------------------- | ------------------- | ---------------------------- |
| **RTO** (Recovery Time Objective)  | 24 hour             | ⚠️ Manual failover           |
| **RPO** (Recovery Point Objective) | 2 hours             | ✅ Automated 2-hourly backups |
| **Data Consistency Check**         | After every restore | 📋 In progress               |
| **Team DR Drill**                  | Quarterly           | 📋 Planned                   |

---

## Incident Severity Levels

| Level        | Definition                                    | Response Time | Escalation                |
| ------------ | --------------------------------------------- | ------------- | ------------------------- |
| **Critical** | All services down; all users affected         | 15 minutes    | On-call → Tech Lead → CTO |
| **High**     | >10% services down; <50% users affected       | 30 minutes    | On-call → Tech Lead       |
| **Medium**   | Partial degradation; <10% users affected      | 4 hours       | Tech Lead                 |
| **Low**      | Non-critical feature broken; monitoring alert | 24 hours      | Standard ticket process   |

---

## Runbook: Belgium Database Fails (Replica Still Healthy)

### Scenario
- Belgium MySQL primary is down (crashed, hardware failure, etc.)
- USA MySQL replica is healthy and up-to-date
- Users see connection timeouts

### Detection
- Prometheus: `mysql_up{instance="db-primary"}` = 0
- AlertManager fires: Database replica lag metric spikes
- Users report "service unavailable"

### Response Steps (Estimated 30 minutes)

#### Step 1: Assess Replica Status (5 min)
```bash
# SSH to USA region
ssh ops@usa-api-server

# Check replica health
docker exec ai-judge-mysql-db mysql -u root -p${MYSQL_ROOT_PASSWORD} -e "SHOW SLAVE STATUS\G" \
  | grep -E "Slave_IO_Running|Slave_SQL_Running|Seconds_Behind_Master"

# Expected output (replica indicates streaming is working):
# Slave_IO_Running: Yes      (IO thread receiving binlog)
# Slave_SQL_Running: Yes     (SQL thread applying changes)
# Seconds_Behind_Master: 0   (or low number like 1-2)

# If all green, replica is ready to be promoted to primary
```

#### Step 2: Stop Replication (2 min)
```bash
# Stop SQL thread to make it read-only
docker exec ai-judge-mysql-db mysql -u root -p${MYSQL_ROOT_PASSWORD} << EOF
STOP SLAVE;
EOF

# Verify replication is stopped
docker exec ai-judge-mysql-db mysql -u root -p${MYSQL_ROOT_PASSWORD} \
  -e "SHOW SLAVE STATUS\G" | head -20
```

#### Step 3: Promote Replica to Primary (2 min)
```bash
# Remove replica configuration and make it writable
docker exec ai-judge-mysql-db mysql -u root -p${MYSQL_ROOT_PASSWORD} << EOF
RESET SLAVE ALL;
SET GLOBAL read_only = OFF;
EOF

# Verify it's now writable (primary)
docker exec ai-judge-mysql-db mysql -u root -p${MYSQL_ROOT_PASSWORD} \
  -e "SHOW VARIABLES LIKE 'read_only';"
```

#### Step 4: Update API Configuration (3 min)
```bash
# On USA API servers, update .env
echo "MYSQL_HOST=localhost" >> .env

# Restart API
docker compose restart api

# Check health
curl -s http://localhost:5555/health | jq .
```

#### Step 5: Update Load Balancer DNS (3 min)
```bash
# Update DNS A record to route to USA only
aws route53 change-resource-record-sets \
  --hosted-zone-id Z1234567890 \
  --change-batch '{
    "Changes": [{
      "Action": "UPSERT",
      "ResourceRecordSet": {
        "Name": "api.aijudge.example.com",
        "Type": "A",
        "TTL": 300,
        "ResourceRecords": [
          {"Value": "34.56.7.89"}
        ]
      }
    }]
  }'
```

#### Step 6: Monitor & Verify (5 min)
```bash
# Check API health
curl https://api.aijudge.example.com/health

# Verify metrics stable in Prometheus Dashboard
# Announce in Slack: "Failed over to USA. Belgium primary offline. Services running normally."
```

---

## Runbook: Both Databases Lost

### Scenario
- Belgium AND USA databases crashed simultaneously
- Video storage intact (backups exist)
- RTO: 2+ hours (critical incident)

### Response Steps (Estimated 2+ hours)

#### Step 1: Locate Latest Backup (5 min)
```bash
# List S3 backups (newest first)
aws s3 ls s3://judge-backups/databases/ --recursive | sort -r | head -5
# Output: judge_db_20260314_120000.sql.enc (6-hour-old backup ✅)

# Assess data loss
echo "**WARNING**: Latest backup is 6 hours old. Data from last 6 hours lost."
```

#### Step 2: Verify Backup Integrity (10 min)
```bash
# Download and decrypt
aws s3 cp s3://judge-backups/databases/judge_db_20260314_120000.sql.enc /tmp/

openssl enc -aes-256-cbc -d -in /tmp/judge_db_20260314_120000.sql.enc \
  -out /tmp/judge_db_backup.sql \
  -pass pass:${BACKUP_PASSWORD}

# Verify valid SQL
head -50 /tmp/judge_db_backup.sql | grep -E "^CREATE DATABASE|^USE judge"

# Check file size (should be large, e.g., 100MB+)
ls -lh /tmp/judge_db_backup.sql
```

#### Step 3: Spin Up New MySQL Instance (15 min)
```bash
# Recover to Belgium first (or USA, decide quickly)
cd /home/judge/
docker compose -f docker-compose.yaml -f docker-compose.prod.yaml up -d mysqldb

sleep 30

# Check health
docker exec ai-judge-mysql-db mysql -u root -p${MYSQL_ROOT_PASSWORD} -e "SELECT 1"
```

#### Step 4: Restore Database (30+ min)
```bash
# Restore
cat /tmp/judge_db_backup.sql | docker exec -i ai-judge-mysql-db \
  mysql -u root -p${MYSQL_ROOT_PASSWORD} judge_db

# Verify
docker exec ai-judge-mysql-db mysql -u root -p${MYSQL_ROOT_PASSWORD} \
  -e "SELECT COUNT(*) as video_count FROM Videos;"

docker exec ai-judge-mysql-db mysql -u root -p${MYSQL_ROOT_PASSWORD} \
  -e "SELECT COUNT(*) as account_count FROM Accounts;"
```

#### Step 5: Restart API Services (10 min)
```bash
docker compose -f docker-compose.yaml -f docker-compose.prod.yaml up -d api web

sleep 30

curl http://localhost:5555/auth/me  # Should require auth, not 500
```

#### Step 6: Validate Data Integrity (15 min)
```bash
docker exec ai-judge-mysql-db mysql -u root -p${MYSQL_ROOT_PASSWORD} judge_db << EOF
-- Check for orphaned records
SELECT COUNT(*) FROM Videos v WHERE NOT EXISTS (
  SELECT 1 FROM Accounts a WHERE a.id = v.owner_id
);
-- Should return 0
EOF
```

#### Step 7: Notify Users (ASAP)
```
📢 Status Page:
"Database failure occurred 11:45-1:30 PM UTC. Services now restored.
Data loss (~6 hours) recovered from backup. Contact support if issues."
```

---

## Runbook: Video Storage Corrupted / Lost

### Scenario
- Videos in DB but files missing/corrupted
- Users can't download or view videos

### Response Steps (30 min to 2+ hours)

#### Step 1: Assess Scope (10 min)
```bash
# How many videos affected?
docker exec ai-judge-api find /storage/videos -type f | wc -l

# Compare with database
docker exec ai-judge-mysql-db mysql -u root -p${MYSQL_ROOT_PASSWORD} \
  judge_db -e "SELECT COUNT(*) FROM Videos;"
```

#### Step 2: Restore Videos from Backup (30+ min)
```bash
# From S3
aws s3 sync s3://judge-backups/videos/ /storage/videos/ --delete

# From rsync backup
rsync -avz /backups/videos/latest-sync/ /storage/videos/
```

#### Step 3: Verify (10 min)
```bash
# Spot-check a few videos
for video in $(docker exec ai-judge-api ls /storage/videos/ | head -5); do
  if [ -f "/storage/videos/$video" ]; then
    echo "✅ $video"
  else
    echo "❌ $video MISSING"
  fi
done
```

---

## Quarterly Disaster Recovery Drill

### Schedule: Last Friday of each quarter

### Drill Steps (2-3 hours)

1. **Announce Exercise** (5 min)
   - "This is a DRILL: Database failover exercise starts now"
   - Post in Slack #incident-response

2. **Simulate Belgium Database Failure** (10 min)
   - Stop Belgium MySQL: `docker compose stop mysqldb`
   - Verify users see errors
   - Check Prometheus alert fires

3. **Execute Failover Runbook** (45-60 min)
   - Promote USA replica to primary
   - Update DNS/load balancer
   - Restart API services
   - **Measure time to recovery**

4. **Validate Services** (15 min)
   - Test user logins
   - Spot-check video access
   - Verify no data corruption

5. **Document Issues** (15 min)
   - Any outdated runbook steps?
   - Missing monitoring?

6. **Restore Original State** (20 min)
   - Bring Belgium back online
   - Reconfigure as primary + replica

7. **Post-Drill Review** (30 min)
   - Team meeting: what went well? what failed?
   - Update runbooks if needed

---

## Contact & Escalation

### On-Call Rotation
- **Weeks 1-13**: @alice (Belgium timezone)
- **Weeks 14-26**: @bob (USA timezone)
- Rotate every quarter

### Escalation Path
1. On-call engineer: Initial response (15 min)
2. Tech Lead: Engaged if RTO > 30 min
3. CTO: Engaged if all services down
4. CEO: Engaged if RTO > 2 hours or data loss

### Communication During Incident
- Slack: #incident-response (internal only)
- Status Page: public.aijudge.example.com/status (for users)
- Email: ops@aijudge.example.com (for urgent reports)

---

## Weekly & Monthly Checklists

### Weekly (Every Monday)
- [ ] Check Prometheus metrics (latency, errors, disk usage)
- [ ] Review AlertManager alerts (should be few/zero false positives)
- [ ] Verify both regions API health endpoints return 200
- [ ] Check database replica lag (should be < 1 second)
- [ ] Verify latest backup exists and is recent (< 6 hours old)

### Monthly (First of month)
- [ ] Review access logs for suspicious activity
- [ ] Test restore from backup (in staging environment)
- [ ] Update runbooks if new issues discovered
- [ ] Check SSL certificate expiration dates
- [ ] Review disk usage trends (growing too fast?)

### Quarterly (Last week of quarter)
- [ ] Execute full DR drill (see above)
- [ ] Team training on new procedures
- [ ] Capacity planning review
- [ ] Security audit (basic)

