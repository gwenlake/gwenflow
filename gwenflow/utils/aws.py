import json

try:
    import boto3
except ImportError:
    raise ImportError("Please install boto3 with `pip install boto3`.")


def aws_get_instance_ip_address(instance_id, region_name="eu-west-3"):
    ec2 = boto3.resource('ec2', region_name=region_name)
    instance = ec2.Instance(instance_id)
    return instance.private_ip_address

def aws_s3_list_files(bucket: str, prefix: str, recursive: bool = True):
    files = []
    client = boto3.client('s3')
    if recursive:
        for key in client.list_objects(Bucket=bucket, Prefix=prefix)['Contents']:
            files.append(key['Key'])
    else:
        for key in client.list_objects(Bucket=bucket, Prefix=prefix, Delimiter='/')['CommonPrefixes']:
            files.append(key['Prefix'])
    return files

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
