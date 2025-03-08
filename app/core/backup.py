import boto3
import json
import logging
import os
from datetime import datetime
import tempfile
import tarfile
import subprocess
from typing import Optional, Dict, Any, List
from app.core.config import settings
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

logger = logging.getLogger(__name__)

class BackupManager:
    """Manager for database and configuration backups"""
    
    def __init__(self, bucket_name: Optional[str] = None):
        """Initialize the backup manager"""
        self.bucket_name = bucket_name or settings.AWS_S3_BUCKET
        self.s3_client = boto3.client(
            's3',
            region_name=settings.AWS_REGION
        )
    
    def backup_database(self, db_name: str, db_user: str, db_password: str, db_host: str) -> bool:
        """Create a backup of the database and upload to S3"""
        try:
            # Create timestamp for the backup
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            backup_filename = f"db_backup_{db_name}_{timestamp}.sql"
            temp_dir = tempfile.gettempdir()
            backup_filepath = os.path.join(temp_dir, backup_filename)
            
            # Create dump using pg_dump
            cmd = [
                "pg_dump",
                f"--host={db_host}",
                f"--username={db_user}",
                "--format=plain",
                f"--file={backup_filepath}",
                db_name
            ]
            
            # Set password in environment
            env = os.environ.copy()
            env["PGPASSWORD"] = db_password
            
            # Execute pg_dump
            logger.info(f"Creating database backup for {db_name}")
            process = subprocess.run(cmd, env=env, capture_output=True, text=True)
            
            if process.returncode != 0:
                logger.error(f"Database backup failed: {process.stderr}")
                return False
            
            # Upload to S3
            s3_key = f"backups/database/{backup_filename}"
            logger.info(f"Uploading database backup to S3: {s3_key}")
            self.s3_client.upload_file(backup_filepath, self.bucket_name, s3_key)
            
            # Clean up temp file
            os.remove(backup_filepath)
            
            logger.info(f"Database backup completed successfully: {s3_key}")
            return True
            
        except Exception as e:
            logger.error(f"Error during database backup: {str(e)}")
            return False
    
    def backup_config(self, config_data: Dict[str, Any]) -> bool:
        """Backup configuration to S3"""
        try:
            # Create timestamp for the backup
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            config_filename = f"config_backup_{timestamp}.json"
            
            # Convert config to JSON and save to temp file
            temp_dir = tempfile.gettempdir()
            config_filepath = os.path.join(temp_dir, config_filename)
            
            with open(config_filepath, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            # Upload to S3
            s3_key = f"backups/config/{config_filename}"
            logger.info(f"Uploading config backup to S3: {s3_key}")
            self.s3_client.upload_file(config_filepath, self.bucket_name, s3_key)
            
            # Clean up temp file
            os.remove(config_filepath)
            
            logger.info(f"Config backup completed successfully: {s3_key}")
            return True
            
        except Exception as e:
            logger.error(f"Error during config backup: {str(e)}")
            return False
    
    def list_backups(self, prefix: str = "backups/") -> List[Dict[str, Any]]:
        """List all available backups"""
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )
            
            backups = []
            if 'Contents' in response:
                for item in response['Contents']:
                    backups.append({
                        'key': item['Key'],
                        'size': item['Size'],
                        'last_modified': item['LastModified'].isoformat(),
                        'type': 'database' if 'database' in item['Key'] else 'config'
                    })
            
            return backups
            
        except Exception as e:
            logger.error(f"Error listing backups: {str(e)}")
            return []
    
    def restore_database(self, backup_key: str, db_name: str, db_user: str, db_password: str, db_host: str) -> bool:
        """Restore database from a backup"""
        try:
            # Download backup from S3
            temp_dir = tempfile.gettempdir()
            backup_filename = os.path.basename(backup_key)
            backup_filepath = os.path.join(temp_dir, backup_filename)
            
            logger.info(f"Downloading database backup from S3: {backup_key}")
            self.s3_client.download_file(self.bucket_name, backup_key, backup_filepath)
            
            # Connect to PostgreSQL to drop and recreate the database
            conn = psycopg2.connect(
                user=db_user,
                password=db_password,
                host=db_host,
                database="postgres"  # Connect to default DB for admin operations
            )
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            cursor = conn.cursor()
            
            # Drop existing database if it exists
            logger.info(f"Dropping existing database {db_name} if it exists")
            cursor.execute(f"DROP DATABASE IF EXISTS {db_name};")
            
            # Create fresh database
            logger.info(f"Creating new database {db_name}")
            cursor.execute(f"CREATE DATABASE {db_name};")
            
            # Close admin connection
            cursor.close()
            conn.close()
            
            # Restore from backup
            cmd = [
                "psql",
                f"--host={db_host}",
                f"--username={db_user}",
                f"--dbname={db_name}",
                f"--file={backup_filepath}"
            ]
            
            # Set password in environment
            env = os.environ.copy()
            env["PGPASSWORD"] = db_password
            
            # Execute psql restore
            logger.info(f"Restoring database from backup: {backup_key}")
            process = subprocess.run(cmd, env=env, capture_output=True, text=True)
            
            if process.returncode != 0:
                logger.error(f"Database restore failed: {process.stderr}")
                return False
            
            # Clean up temp file
            os.remove(backup_filepath)
            
            logger.info(f"Database restore completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error during database restore: {str(e)}")
            return False
    
    def restore_config(self, backup_key: str) -> Optional[Dict[str, Any]]:
        """Restore configuration from a backup"""
        try:
            # Download backup from S3
            temp_dir = tempfile.gettempdir()
            config_filename = os.path.basename(backup_key)
            config_filepath = os.path.join(temp_dir, config_filename)
            
            logger.info(f"Downloading config backup from S3: {backup_key}")
            self.s3_client.download_file(self.bucket_name, backup_key, config_filepath)
            
            # Load config from JSON
            with open(config_filepath, 'r') as f:
                config_data = json.load(f)
            
            # Clean up temp file
            os.remove(config_filepath)
            
            logger.info(f"Config restore completed successfully")
            return config_data
            
        except Exception as e:
            logger.error(f"Error during config restore: {str(e)}")
            return None
