"""HTTPS webhook transport with connection-time address pinning and bounded I/O."""

import hashlib
import hmac
import http.client
import ipaddress
import socket
import ssl
from dataclasses import dataclass
from urllib.parse import urljoin, urlsplit


@dataclass(frozen=True, slots=True)
class Destination:
    host: str
    port: int
    path: str


def validate_destination(url: str) -> Destination:
    if len(url) > 2048 or any(ord(c) <= 32 or ord(c) == 127 for c in url):
        raise ValueError("Invalid webhook URL")
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
    ):
        raise ValueError("Webhook destination must be credential-free HTTPS")
    host = parsed.hostname.encode("idna").decode("ascii").lower()
    port = parsed.port or 443
    if port != 443:
        raise ValueError("Webhook policy permits HTTPS port 443 only")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        if "." not in host or host.endswith("."):
            raise ValueError("Webhook hostname must be fully qualified") from None
    else:
        require_public(str(address))
    return Destination(
        host, port, (parsed.path or "/") + ("?" + parsed.query if parsed.query else "")
    )


def require_public(value: str) -> None:
    address = ipaddress.ip_address(value)
    effective = address.ipv4_mapped if isinstance(address, ipaddress.IPv6Address) else None
    candidate = effective or address
    if (
        not candidate.is_global
        or candidate.is_multicast
        or candidate.is_loopback
        or candidate.is_link_local
        or candidate.is_reserved
        or candidate.is_unspecified
        or "%" in value
    ):
        raise ValueError("Forbidden webhook network destination")
    # Reject transition mechanisms that can conceal a distinct IPv4 destination.
    if isinstance(address, ipaddress.IPv6Address):
        if address.sixtofour is not None or address.teredo is not None:
            raise ValueError("IPv6 transition destinations are not permitted")
        if address in ipaddress.ip_network("64:ff9b::/96"):
            require_public(str(ipaddress.IPv4Address(int(address) & 0xFFFFFFFF)))


class PinnedHTTPSConnection(http.client.HTTPSConnection):
    """Resolve once, validate every answer, connect numerically, preserve TLS SNI."""

    def connect(self) -> None:
        addresses = socket.getaddrinfo(self.host, self.port, type=socket.SOCK_STREAM)
        if not addresses:
            raise OSError("Webhook DNS returned no addresses")
        for _, _, _, _, sockaddr in addresses:
            require_public(str(sockaddr[0]))
        family, kind, protocol, _, sockaddr = addresses[0]
        raw = socket.socket(family, kind, protocol)
        try:
            raw.settimeout(3.0)
            raw.connect(sockaddr)
            self.sock = self._context.wrap_socket(raw, server_hostname=self.host)  # type: ignore[attr-defined]
            self.sock.settimeout(10.0)
        except BaseException:
            raw.close()
            raise


def deliver(
    url: str,
    body: bytes,
    secret: bytes,
    delivery_id: str,
    timestamp: int,
    max_payload_bytes: int | None = None,
    connect_timeout: float | None = None,
    max_response_bytes: int | None = None,
    max_redirects: int | None = None,
) -> int:
    from src.config.settings import get_settings

    cfg = get_settings().external
    max_payload = (
        max_payload_bytes if max_payload_bytes is not None else cfg.webhook_max_payload_bytes
    )
    conn_to = (
        connect_timeout if connect_timeout is not None else cfg.webhook_connect_timeout_seconds
    )
    max_resp = (
        max_response_bytes if max_response_bytes is not None else cfg.webhook_max_response_bytes
    )
    max_redir = max_redirects if max_redirects is not None else cfg.webhook_max_redirects

    if not secret or len(body) > max_payload or timestamp < 0:
        raise ValueError("Invalid webhook signing input or payload")
    if (
        not delivery_id
        or len(delivery_id) > 128
        or not all(c.isalnum() or c in "-_" for c in delivery_id)
    ):
        raise ValueError("Invalid delivery identifier")
    original = validate_destination(url)
    signature = hmac.new(secret, str(timestamp).encode() + b"." + body, hashlib.sha256).hexdigest()
    current = url
    for redirect in range(max_redir):
        destination = validate_destination(current)
        if (destination.host, destination.port) != (original.host, original.port):
            raise ValueError("Cross-origin webhook redirects are not permitted")
        connection = PinnedHTTPSConnection(
            destination.host,
            destination.port,
            timeout=conn_to,
            context=ssl.create_default_context(),
        )
        try:
            connection.request(
                "POST",
                destination.path,
                body=body,
                headers={
                    "Content-Type": "application/json",
                    "X-Growth-Delivery-ID": delivery_id,
                    "X-Growth-Timestamp": str(timestamp),
                    "X-Growth-Signature": signature,
                },
            )
            response = connection.getresponse()
            if len(response.read(max_resp + 1)) > max_resp:
                raise ValueError("Oversized webhook response")
            if response.status in (307, 308):
                location = response.getheader("Location")
                if redirect == max_redir - 1 or not location:
                    raise ValueError("Webhook redirect limit or missing destination")
                current = urljoin(current, location)
                continue
            if 300 <= response.status < 400:
                raise ValueError("Webhook redirect would change POST semantics")
            if not 200 <= response.status < 300:
                raise OSError(f"Webhook rejected delivery with HTTP {response.status}")
            return response.status
        finally:
            connection.close()
    raise RuntimeError("Unreachable redirect state")
