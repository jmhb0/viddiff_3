import lmdb
import logging
import os
import shutil
from datetime import datetime
import subprocess
import sys

def recover_chunk(backup_path, new_cache_path, start_key=None, chunk_size=1000):
    """Recover a chunk of entries starting from start_key"""
    recovered = 0
    failed = 0
    
    old_cache = lmdb.open(backup_path, readonly=True)
    new_cache = lmdb.open(new_cache_path, map_size=int(1e12))
    
    try:
        with old_cache.begin() as old_txn, new_cache.begin(write=True) as new_txn:
            cursor = old_txn.cursor()
            
            # Position cursor at start_key if provided
            if start_key:
                cursor.set_key(start_key)
            else:
                cursor.first()
                
            for _ in range(chunk_size):
                try:
                    key, value = cursor.item()
                    new_txn.put(key, value)
                    recovered += 1
                    last_key = key
                    if not cursor.next():
                        break
                except:
                    failed += 1
                    if not cursor.next():
                        break
                    
    except Exception as e:
        logging.error(f"Error processing chunk: {e}")
    finally:
        old_cache.close()
        new_cache.close()
        
    return recovered, failed, last_key if recovered > 0 else None

def recover_lmdb_cache(cache_path="cache/cache_reformat", map_size=int(1e12)):
    """
    Attempts to recover as much data as possible from a corrupted LMDB database.
    """
    backup_path = f"{cache_path}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    new_cache_path = f"{cache_path}_new"

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # First make a backup of the corrupted cache
    if os.path.exists(cache_path):
        logging.info(f"Creating backup of corrupted cache at {backup_path}")
        shutil.copytree(cache_path, backup_path)
    else:
        logging.error(f"Cache path {cache_path} does not exist!")
        return 0, 0

    total_recovered = 0
    total_failed = 0
    last_key = None
    chunk_size = 1000

    while True:
        # Run recovery chunk in subprocess to handle crashes
        cmd = [
            sys.executable, __file__,
            '--recover-chunk',
            '--backup-path', backup_path,
            '--new-cache-path', new_cache_path,
            '--chunk-size', str(chunk_size)
        ]
        if last_key:
            cmd.extend(['--start-key', last_key.decode('utf-8')])
            
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logging.error(f"Chunk recovery failed: {result.stderr}")
                break
                
            recovered, failed, new_last_key = eval(result.stdout)
            if new_last_key:
                last_key = new_last_key.encode('utf-8')
            else:
                break  # No more entries to process
                
            total_recovered += recovered
            total_failed += failed
            
            logging.info(f"Recovered {total_recovered} entries so far...")
            
        except Exception as e:
            logging.error(f"Error running recovery subprocess: {e}")
            
            
            break

    # Replace old cache with new one
    if os.path.exists(cache_path):
        shutil.rmtree(cache_path)
    shutil.move(new_cache_path, cache_path)

    logging.info(f"Recovery complete!")
    logging.info(f"Successfully recovered: {total_recovered} entries")
    logging.info(f"Failed to recover: {total_failed} entries")
    logging.info(f"Original database backed up at: {backup_path}")
    logging.info(f"New database installed at: {cache_path}")

    return total_recovered, total_failed

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Recover corrupted LMDB database')
    parser.add_argument('--cache-path', 
                       default="cache/cache_openai",
                       help='Path to the corrupted LMDB database')
    parser.add_argument('--map-size', 
                       type=int,
                       default=int(1e12),
                       help='Size of the new database map')
    parser.add_argument('--recover-chunk', action='store_true',
                       help='Internal use: recover a chunk of the database')
    parser.add_argument('--backup-path', help='Path to backup database')
    parser.add_argument('--new-cache-path', help='Path to new database')
    parser.add_argument('--start-key', help='Start key for chunk recovery')
    parser.add_argument('--chunk-size', type=int, help='Size of chunk to recover')
    
    args = parser.parse_args()
    
    if args.recover_chunk:
        # Running as subprocess to recover a chunk
        recovered, failed, last_key = recover_chunk(
            args.backup_path,
            args.new_cache_path,
            args.start_key.encode('utf-8') if args.start_key else None,
            args.chunk_size
        )
        print(repr((recovered, failed, last_key.decode('utf-8') if last_key else None)))
        exit(0)
    
    # Main recovery process
    recovered, failed = recover_lmdb_cache(args.cache_path, args.map_size)
    
    if recovered + failed == 0:
        exit(1)
    exit(0) 