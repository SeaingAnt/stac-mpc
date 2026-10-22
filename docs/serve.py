#!/usr/bin/env python3
"""
Lightweight development HTTP server for docs/ with HTTP Range (206 Partial Content) support.
Handles video streaming cleanly and suppresses benign client-side BrokenPipeErrors.
"""

import os
import sys
from http import HTTPStatus
from http.server import HTTPServer, SimpleHTTPRequestHandler


class VideoFriendlyHTTPRequestHandler(SimpleHTTPRequestHandler):
    """SimpleHTTPRequestHandler with HTTP Range (byte-range) support for video streaming."""

    def copyfile(self, source, outputfile):
        """Copy file buffer, ignoring BrokenPipe / ConnectionReset when client disconnects or seeks."""
        try:
            super().copyfile(source, outputfile)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def handle_one_request(self):
        try:
            super().handle_one_request()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, format, *args):
        # Clean logging: filter out harmless 499/client disconnects
        sys.stderr.write("%s - - [%s] %s\n" %
                         (self.address_string(),
                          self.log_date_time_string(),
                          format % args))

    def end_headers(self):
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def send_head(self):
        path = self.translate_path(self.path)
        if os.path.isdir(path):
            return super().send_head()

        range_header = self.headers.get("Range")
        if not range_header or not os.path.exists(path):
            return super().send_head()

        try:
            f = open(path, "rb")
        except OSError:
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return None

        fs = os.fstat(f.fileno())
        file_size = fs.st_size

        if not range_header.startswith("bytes="):
            f.close()
            return super().send_head()

        range_str = range_header[6:].strip()
        parts = range_str.split("-")
        try:
            if parts[0]:
                start = int(parts[0])
                end = int(parts[1]) if parts[1] else file_size - 1
            else:
                suffix = int(parts[1])
                start = max(0, file_size - suffix)
                end = file_size - 1
        except ValueError:
            f.close()
            self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            return None

        if start >= file_size or end >= file_size or start > end:
            f.close()
            self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            self.send_header("Content-Range", f"bytes */{file_size}")
            self.end_headers()
            return None

        length = end - start + 1
        self.send_response(HTTPStatus.PARTIAL_CONTENT)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()

        f.seek(start)

        class FileRangeWrapper:
            def __init__(self, fp, length):
                self.fp = fp
                self.remaining = length

            def read(self, size=-1):
                if self.remaining <= 0:
                    return b""
                if size < 0 or size > self.remaining:
                    size = self.remaining
                data = self.fp.read(size)
                self.remaining -= len(data)
                return data

            def close(self):
                self.fp.close()

        return FileRangeWrapper(f, length)


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    docs_dir = os.path.abspath(os.path.dirname(__file__))
    os.chdir(docs_dir)

    server = HTTPServer(("0.0.0.0", port), VideoFriendlyHTTPRequestHandler)
    print(f"Serving docs at http://localhost:{port}/ (Press Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")


if __name__ == "__main__":
    main()
