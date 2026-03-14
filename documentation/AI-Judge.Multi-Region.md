# Multi-Region Infrastructure for Belgium & USA

**Goal**: Deploy AI Judge to USA and Belgium with redundancy, failover, and data consistency.

**See also**: [[AI-Judge.Roadmap]] (Phase 2), [[AI-Judge.Disaster-Recovery]], [[AI-Judge.Backup]]

---

## Architecture Overview

```
┌─────────────────────┐                    ┌─────────────────────┐
│   BELGIUM (EU)      │                    │     USA             │
│                     │                    │                     │
├─ App Server 1      │                    ├─ App Server 1      │
├─ MySQL Primary     │◄──Replication──►├─ MySQL Replica     │
├─ Video Storage     │◄──S3/Rsync─────►├─ Video Storage     │
└─────────────────────┘                    └─────────────────────┘
         ▲                                          ▲
         │                                         │
         └──────────────────┬──────────────────────┘
                            │
                      Load Balancer
                      (Nginx or Cloud)
                            │
                      ┌─────────────┐
                      │  End Users  │
                      └─────────────┘
```

### Key Principles

1. **MySQL Replication**: Belgium (primary) → USA (replica)
2. **Automatic Health Checks**: Each region monitors its services
3. **DNS Failover** (manual initially, automated later)
4. **Data Sync**: Videos/generated data replicate via S3-compatible storage or rsync
5. **Secrets Management**: Each region stores secrets locally; no cross-region sync
6. **Monitoring**: Region health reported centrally; alerts trigger failover decision

---

## 1. Database Replication Strategy

### MySQL Replication Setup (Primary → Replica)

#### Primary Instance (Belgium)

Configure binary logging to enable replication:

```yaml
# docker-compose.yaml - Belgium
services:
  mysqldb-primary:
    image: mysql:8.4
    environment:
      MYSQL_ROOT_PASSWORD: ${MYSQL_ROOT_PASSWORD}
      MYSQL_DATABASE: ${MYSQL_DATABASE}
    command: >
      --server-id=1
      --log-bin=mysql-bin
      --binlog-format=ROW
      --binlog-row-image=FULL
      --skip-name-resolve
    ports:
      - "3306:3306"
    volumes:
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
      - mysql_data_primary:/var/lib/mysql
    healthcheck:
      test: ["CMD", "mysqladmin", "ping", "-h", "localhost"]
      interval: 10s
      retries: 5

volumes:
  mysql_data_primary:
```

#### Replica Instance (USA)

```yaml
# docker-compose.yaml - USA
services:
  mysqldb-replica:
    image: mysql:8.4
    environment:
      MYSQL_ROOT_PASSWORD: ${MYSQL_ROOT_PASSWORD}
      MYSQL_DATABASE: ${MYSQL_DATABASE}
    command: >
      --server-id=2
      --skip-name-resolve
      --relay-log=relay-bin
      --relay-log-index=relay-bin.index
      --relay-log-recovery=ON
    ports:
      - "3306:3306"
    depends_on:
      mysqldb-primary:
        condition: service_healthy
    volumes:
      - mysql_data_replica:/var/lib/mysql
    healthcheck:
      test: ["CMD", "mysqladmin", "ping", "-h", "localhost"]
      interval: 10s
      retries: 5

volumes:
  mysql_data_replica:
```

### Setup Replication (One-time)

#### Step 1: Create replication user on primary

```sql
-- Run on PRIMARY (Belgium)
CREATE USER 'replication'@'%' IDENTIFIED BY '${REPLICATION_PASSWORD}';
GRANT REPLICATION SLAVE, REPLICATION CLIENT ON *.* TO 'replication'@'%';
FLUSH PRIVILEGES;
```

#### Step 2: Get primary binlog position

```bash
# Run on PRIMARY container
mysql -h localhost -u root -p${MYSQL_ROOT_PASSWORD} -e "SHOW MASTER STATUS;"
# Output:
# File | Position
# mysql-bin.000001 | 154
```

#### Step 3: Configure replica

```bash
# Run on REPLICA container
mysql -h localhost -u root -p${MYSQL_ROOT_PASSWORD} << EOF
CHANGE MASTER TO
MASTER_HOST='mysqldb-primary',
MASTER_USER='replication',
MASTER_PASSWORD='${REPLICATION_PASSWORD}',
MASTER_LOG_FILE='mysql-bin.000001',
MASTER_LOG_POS=154;

START SLAVE;
SHOW SLAVE STATUS\G
EOF
```

#### Step 4: Verify replication

```bash
# On replica, check Seconds_Behind_Master (should be ~0 or low number)
mysql -h localhost -u root -p${MYSQL_ROOT_PASSWORD} -e "SHOW SLAVE STATUS\G" | grep Seconds_Behind_Master
```

### Monitoring Replica Lag

Add health endpoint in API (see [[AI-Judge.Disaster-Recovery]] for detailed runbooks):

```python
# api/routers/healthRouter.py
@bp.route('/health/database-replica-lag', methods=['GET'])
def database_replica_lag():
    """Return replica lag in seconds (or None if primary)."""
    try:
        result = db.session.execute("SHOW SLAVE STATUS").fetchone()
        if result and result.get('Seconds_Behind_Master') is not None:
            lag = result['Seconds_Behind_Master']
            status_code = 200 if lag < 5 else 503
            return jsonify({"replica_lag_seconds": lag}), status_code
        else:
            # Primary node (no replica)
            return jsonify({"replica_lag_seconds": None, "is_primary": True}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
```

---

## 2. Video & Generated Data Replication

Videos and generated data are replicated using **rsync-based bidirectional sync**.
The sync process is managed by the sync-executor service (see [[AI-Judge.Bidirectional-Sync]]).

### Video Replication Process

- **Trigger**: When videos are labeled (FrameLabel records created)
- **Method**: Database-driven selection + rsync over SSH
- **Frequency**: Continuously via sync-executor (runs every 5 minutes)
- **Direction**: Bidirectional (any region can be source)
- **Scope**: Only videos with frame labels are synced

See [[AI-Judge.Bidirectional-Sync]] for complete setup.

---

## 3. Secrets Management

Store region-specific secrets:

**Belgium (.env-be)**
```env
REGION=belgium
MYSQL_SERVER_ID=1
MYSQL_HOST=mysqldb-be
ALL_REGIONS=belgium,usa
REGION_BELGIUM_HOST=sync-belgium.vpn
REGION_USA_HOST=sync-usa.vpn
```

**USA (.env-us)**
```env
REGION=usa
MYSQL_SERVER_ID=2
MYSQL_HOST=mysqldb-us
ALL_REGIONS=belgium,usa
REGION_BELGIUM_HOST=sync-belgium.vpn
REGION_USA_HOST=sync-usa.vpn
```

---

## 4. Deployment Checklist

- [ ] MySQL replication configured and verified (Belgium → USA)
- [ ] Replica lag < 5s under normal load
- [ ] SSH keys configured for cross-region rsync (sync-executor)
- [ ] Video sync (rsync) running automatically via sync-executor
- [ ] Backups automated and encrypted
- [ ] Team trained on failover steps (see [[AI-Judge.Disaster-Recovery]])
- [ ] SSL certs valid for both regions (managed by website DNS)
- [ ] Monitor replication lag via `/health/database-replica-lag` endpoint

---

## For Detailed Operations

See [[AI-Judge.Disaster-Recovery]] for:
- Failover runbooks (Belgium DB fails)
- Data recovery scenarios
- Team escalation paths
- Quarterly DR drills

