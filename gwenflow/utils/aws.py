import io
import os
import json
from collections import namedtuple
from operator import attrgetter

try:
    import boto3
except ImportError:
    raise ImportError("Please install boto3 with `pip install boto3`.")


def aws_get_instance_ip_address(instance_id, region_name="eu-west-3"):
    ec2 = boto3.resource('ec2', region_name=region_name)
    instance = ec2.Instance(instance_id)
    return instance.private_ip_address

def __prev_str(s):
    if len(s) == 0:
        return s
    s, c = s[:-1], ord(s[-1])
    if c > 0:
        s += chr(c - 1)
    s += ''.join(['\u7FFF' for _ in range(10)])
    return s

# Thanks to https://stackoverflow.com/questions/35803027/retrieving-subfolders-names-in-s3-bucket-from-boto3
s3object = namedtuple('S3Obj', ['key', 'mtime', 'size', 'ETag'])

def aws_s3_list_files(bucket, path, start=None, end=None, recursive=True, list_dirs=True,
           list_objs=True, limit=None):
    """
    Iterator that lists a bucket's objects under path, (optionally) starting with
    start and ending before end.

    If recursive is False, then list only the "depth=0" items (dirs and objects).

    If recursive is True, then list recursively all objects (no dirs).

    Args:
        bucket:
            a boto3.resource('s3').Bucket().
        path:
            a directory in the bucket.
        start:
            optional: start key, inclusive (may be a relative path under path, or
            absolute in the bucket)
        end:
            optional: stop key, exclusive (may be a relative path under path, or
            absolute in the bucket)
        recursive:
            optional, default True. If True, lists only objects. If False, lists
            only depth 0 "directories" and objects.
        list_dirs:
            optional, default True. Has no effect in recursive listing. On
            non-recursive listing, if False, then directories are omitted.
        list_objs:
            optional, default True. If False, then directories are omitted.
        limit:
            optional. If specified, then lists at most this many items.

    Returns:
        an iterator of S3Obj.

    Examples:
        # set up
        >>> s3 = boto3.resource('s3')
        ... bucket = s3.Bucket('bucket-name')

        # iterate through all S3 objects under some dir
        >>> for p in aws_s3_list_files(bucket, 'some/dir'):
        ...     print(p)

        # iterate through up to 20 S3 objects under some dir, starting with foo_0010
        >>> for p in aws_s3_list_files(bucket, 'some/dir', limit=20, start='foo_0010'):
        ...     print(p)

        # non-recursive listing under some dir:
        >>> for p in aws_s3_list_files(bucket, 'some/dir', recursive=False):
        ...     print(p)

        # non-recursive listing under some dir, listing only dirs:
        >>> for p in aws_s3_list_files(bucket, 'some/dir', recursive=False, list_objs=False):
        ...     print(p)
    """
    kwargs = dict()
    if start is not None:
        if not start.startswith(path):
            start = os.path.join(path, start)
        # note: need to use a string just smaller than start, because
        # the list_object API specifies that start is excluded (the first
        # result is *after* start).
        kwargs.update(Marker=__prev_str(start))
    if end is not None:
        if not end.startswith(path):
            end = os.path.join(path, end)
    if not recursive:
        kwargs.update(Delimiter='/')
        if not path.endswith('/') and len(path) > 0:
            path += '/'
    kwargs.update(Prefix=path)
    if limit is not None:
        kwargs.update(PaginationConfig={'MaxItems': limit})

    paginator = bucket.meta.client.get_paginator('list_objects')
    for resp in paginator.paginate(Bucket=bucket.name, **kwargs):
        q = []
        if 'CommonPrefixes' in resp and list_dirs:
            q = [s3object(f['Prefix'], None, None, None) for f in resp['CommonPrefixes']]
        if 'Contents' in resp and list_objs:
            q += [s3object(f['Key'], f['LastModified'], f['Size'], f['ETag']) for f in resp['Contents']]
        # note: even with sorted lists, it is faster to sort(a+b)
        # than heapq.merge(a, b) at least up to 10K elements in each list
        q = sorted(q, key=attrgetter('key'))
        if limit is not None:
            q = q[:limit]
            limit -= len(q)
        for p in q:
            if end is not None and p.key >= end:
                return
            yield p

def aws_s3_read_file(bucket: str, file: str):
    client = boto3.client('s3')
    response = client.get_object(Bucket=bucket, Key=file)
    return response['Body'].read()

def aws_s3_delete_folder(bucket: str, prefix: str):
    if len(prefix)<35: # protect repo if prefix is not uuid4
        return False
    client = boto3.client('s3')
    for key in client.list_objects(Bucket=bucket, Prefix=prefix):
        key.delete()
    return True

def aws_s3_read_json_file(bucket: str, file: str):
    data = aws_s3_read_file(bucket=bucket, file=file)
    return json.loads(data.decode('utf-8'))

def aws_s3_read_text_file(bucket: str, file: str):
    data = aws_s3_read_file(bucket=bucket, file=file)
    return data.decode('utf-8')

def aws_s3_uri_to_bucket_key(uri: str):
    try:
        tmp = uri.replace("s3://", "")
        bucket = tmp.split("/")[0]
        key = tmp.replace(bucket, "").strip("/")
        return bucket, key
    except:
        pass
    return None, None

def aws_s3_upload_fileobj(bucket: str, key: str, fileobj, content_type: str = "application/octet-stream") -> dict:
    client = boto3.client('s3')
    try:
        client.upload_fileobj(Fileobj=fileobj,Bucket=bucket,Key=key,ExtraArgs={"ContentType": content_type})
        return {
            "success": True,
            "bucket": bucket,
            "key": key,
            "content_type": content_type
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
    
def aws_s3_delete_file(bucket: str, key: str) -> dict:
    client = boto3.client("s3")
    try:
        client.delete_object(Bucket=bucket, Key=key)
        return {
            "success": True,
            "bucket": bucket,
            "key": key
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
    
def aws_s3_download_in_buffer(bucket: str, key: str):
    try:
        client = boto3.client("s3")
        buffer = io.BytesIO()
        client.download_fileobj(Bucket=bucket, Key=key, Fileobj=buffer)
        buffer.seek(0)
        return buffer
    except Exception as e:
        print(f"Error downloading s3://{bucket}/{key}: {e}")
        return None
    
def aws_s3_is_file(bucket: str, key: str) -> bool:
    client = boto3.client('s3')
    try:
        client.head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False

