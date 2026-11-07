from urllib.parse import urlparse

BLOCKED_HOST_SUFFIXES = (
    "googlesyndication.com",
    "doubleclick.net",
    "googleadservices.com",
    "adtrafficquality.google",
    "google-analytics.com",
    "googletagmanager.com",
    "googletagservices.com",
    "criteo.com",
    "criteo.net",
    "openx.net",
    "adnxs.com",
    "rubiconproject.com",
    "pubmatic.com",
    "amazon-adsystem.com",
    "taboola.com",
    "outbrain.com",
    "adsrvr.org",
    "casalemedia.com",
    "3lift.com",
    "smartadserver.com",
    "moatads.com",
    "scorecardresearch.com",
    "facebook.net",
)


def is_blocked(url):
    host = urlparse(url).hostname or ""
    return any(host == s or host.endswith("." + s) for s in BLOCKED_HOST_SUFFIXES)


def install(context):
    def handle(route):
        if is_blocked(route.request.url):
            route.abort()
        else:
            route.continue_()

    context.route("**/*", handle)
