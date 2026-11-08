"""Multipart upload matching the video form in the web application."""
import json
from urllib.request import Request, urlopen
import uuid


def upload(base, path, start=0, duration=60, compare=True, tolerance=15):
    boundary = uuid.uuid4().hex
    fields = {'start_seconds': start, 'duration_seconds': duration,
              'compare_to_example': str(compare).lower(), 'tolerance_degrees': tolerance}
    body = b''.join((f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n').encode()
                    for key, value in fields.items())
    body += (f'--{boundary}\r\nContent-Disposition: form-data; name="video"; filename="clip.mp4"\r\n'
             'Content-Type: video/mp4\r\n\r\n').encode() + path.read_bytes() + f'\r\n--{boundary}--\r\n'.encode()
    request = Request(f'{base}/api/analyze', data=body,
                      headers={'Content-Type': f'multipart/form-data; boundary={boundary}'})
    with urlopen(request, timeout=600) as response:
        return json.load(response)
