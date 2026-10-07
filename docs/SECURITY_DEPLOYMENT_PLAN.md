# PassShield Security Deployment Plan

## Scope

This plan applies security controls at the hosting, edge, and operations layers. It does not add application source code. The current project is a public Flask application with `POST /analyze`, `POST /improve`, and `POST /export_report`; it has no user accounts, database, or role model. The plan assumes deployment behind a managed edge provider such as Cloudflare and a managed Python host. Translate the settings to the selected provider before rollout.

The repository now includes an Nginx deployment configuration in `deploy/nginx/`. These settings are **not active** until installed in front of the app and verified in the deployment environment. Remaining provider controls are still planned. Keep passwords and password-derived values out of access logs, analytics, traces, and support tickets.

### Render deployment status

The deployed `onrender.com` service uses Render's managed edge, not the repository's Nginx configuration. Render provides managed TLS/HTTP-to-HTTPS redirect ([TLS docs](https://render.com/docs/tls)) and platform DDoS protection ([DDoS docs](https://render.com/docs/ddos-protection)) for web services. Those platform controls are active by default; they do not enable these Nginx endpoint rate limits or custom WAF rules. The current Flask app has no application-level rate limiter. To get configurable endpoint limits without changing this app, put a customer-controlled WAF/CDN in front of a custom domain and prevent bypass through the default Render subdomain after verifying the custom domain ([subdomain controls](https://render.com/docs/custom-domains)). Otherwise, add rate-limiting middleware to the application.

### Repository configuration implemented

- `deploy/nginx/http-security.conf` defines per-client request limit zones and hides the Nginx version banner.
- `deploy/nginx/passshield-site.conf` redirects HTTP to HTTPS, sets TLS and security headers, disables caching, limits request size and connections, and rate-limits `/analyze`, `/export_report`, and `/improve` separately.
- Current limits are 120 requests/minute with burst 10 for `/analyze` and `/export_report`, and 5 requests/minute with burst 2 for `/improve`. Tune after observing real traffic.
- These Nginx zones use `$binary_remote_addr`. If a CDN/reverse proxy sits in front, configure Nginx real-client-IP handling with only that provider's published proxy ranges before enabling rate limits; otherwise many users may share a limit or spoofing may be possible.
- Nginx's default access log format does not record request bodies. Do not enable body logging or body capture in another layer.

## Implementation sequence

### 1. Establish the production boundary

- Serve Flask through the host's production WSGI server; do not expose Flask's development server or debug mode to the Internet.
- Put the application behind a CDN/WAF with TLS enabled and HTTP redirected to HTTPS.
- Restrict direct origin access to the edge provider where the host supports it. If origin restriction is unavailable, use private ingress or authenticated origin access.
- Configure trusted proxy/IP forwarding at the host. Only trust forwarding headers from the edge; otherwise clients can spoof their apparent IP and defeat IP controls.
- Set request body, header, connection, and upstream timeout limits at the edge/host. Reject malformed JSON and unsupported content types at the edge where supported.

### 2. Apply edge abuse controls

- Enable the managed WAF ruleset in logging/monitor mode first. Review false positives, then block confirmed rules.
- Add rate limits to all three POST endpoints. Start with conservative per-IP limits and a short burst allowance, observe legitimate usage, and tune before enforcing. Give `/improve` a tighter limit because it calls a paid external LLM service. Set actual thresholds from expected traffic and provider limits.
- Return `429` with a short `Retry-After` when supported. Ensure the edge does not log request bodies.
- Enable bot/challenge controls only for suspicious traffic patterns so ordinary users and accessibility tools can use the service.
- Use IP reputation feeds and temporary blocks for clear abuse. Geo-block only where there is a business requirement or observed threat; record the reason and expiry date.
- Use provider DDoS protection and CDN caching for static assets. Do not cache password-bearing POST responses or API responses containing analysis results.

### 3. Protect transport, secrets, and data

- Require TLS 1.2 or newer at the edge, enable automatic certificate renewal, and enable HSTS only after HTTPS and subdomains are confirmed to work.
- Keep `GROQ_API_KEY` in the hosting platform's encrypted secret store. Rotate it after suspected exposure and restrict who can view or change production secrets. Do not commit `.env` or print secret values in deployment logs.
- Confirm whether the host persists request bodies or application logs. Disable body capture and redact headers that could contain credentials. Set short retention for operational logs.
- No database or stored user data is present in the current project, so database encryption-at-rest settings do not apply today. Rely on the host's encrypted storage for deployment artifacts and secrets; revisit this section before adding persistent data.
- Review the `/improve` data flow before public deployment: the submitted password is sent to the configured Groq API. Edge protections cannot change that behavior. Do not describe the service as zero-knowledge for this endpoint unless the data flow is changed and verified.

### 4. Authentication and authorization decision

- The current app is intentionally anonymous and has no privileged user actions. OAuth/JWT, RBAC, and MFA are not deployable as meaningful per-user controls without changing the product and application behavior.
- Keep the public endpoints public unless product requirements change. For any future operator-only console or administrative route, put it behind the identity provider's access proxy, require MFA, use least-privilege roles, and keep it separate from anonymous password-analysis routes.
- If account-based features are introduced, define identity, session lifetime, revocation, recovery, and authorization rules before enabling them. A global identity gate would block the current anonymous user flow.

### 5. Resilience and operational visibility

- Configure host health checks, restart policy, resource limits, deployment rollback, and upstream timeouts for Groq and HaveIBeenPwned where supported.
- Use edge/host-level concurrency limits and overload responses to protect Flask workers. Infrastructure settings alone cannot add circuit breakers or reliable retries to Python requests.
- Send access and platform logs to centralized logging. Exclude request bodies and password fields; limit access and define retention.
- Monitor request rate, latency, `4xx`/`5xx` rates, `429` counts, worker restarts, CPU, memory, outbound API failures, and WAF actions. Alert on sustained error/latency increases and unusual spikes, never on password content.
- Use provider DDoS/WAF events as the network intrusion signal. Do not claim application-level anomaly detection unless it is separately implemented and evaluated.

## Rollout and acceptance checklist

1. Install the Nginx fragments in a staging environment: include `http-security.conf` inside the Nginx `http {}` context, install `passshield-site.conf` as a site, replace the example hostname and certificate paths, and point the backend to a private WSGI/Gunicorn listener on `127.0.0.1:8000`.
2. Validate Nginx configuration and reload it; keep any WAF rules in monitor mode first.
3. Confirm HTTPS redirect, valid certificate, origin restriction, and production WSGI serving.
4. Confirm limits return `429` under deliberate request bursts and normal UI use remains functional.
5. Confirm password-bearing requests and responses are absent from logs, analytics, and cache.
6. Confirm `/improve` is rate-limited and its external-provider data flow is disclosed to operators and users.
7. Confirm secrets are unavailable in repository history and deployment logs; rotate any key previously committed or exposed.
8. Verify alerts, log access, retention, incident contacts, and rollback steps with a controlled staging exercise.
9. Promote to production, watch WAF and rate-limit events, and tune thresholds from observed traffic.

Do not mark a control implemented until its provider setting is active and its acceptance check has passed.

## Post-implementation record

After deployment, replace each `TBD` below with the actual configuration and evidence. Do not copy planned controls into a compliance claim before verifying them.

| Control | Provider / setting | Status | Evidence or verification date | Owner / review date |
|---|---|---|---|---|
| Render platform DDoS protection | Render managed edge | Active by platform default; verify service | TBD | TBD |
| Customer-configured CDN WAF and route limits | TBD | Not configured | TBD | TBD |
| TLS and HTTP-to-HTTPS redirect | Render managed TLS | Active by platform default; verify service | TBD | TBD |
| HSTS response header | Nginx template for self-hosting only | Not active on Render | TBD | TBD |
| Origin access restriction | TBD | Not configured | TBD | TBD |
| Rate limits for each POST route | Nginx zones for self-hosting; customer CDN rules or app middleware for Render | Not active on Render | TBD | TBD |
| Bot, reputation, and geographic rules | TBD | Not configured | TBD | TBD |
| Request body and timeout limits | Nginx site config in `deploy/nginx/passshield-site.conf` | Prepared; not active | TBD | TBD |
| Secret storage and rotation | TBD | Not configured | TBD | TBD |
| Log redaction and retention | TBD | Not configured | TBD | TBD |
| Metrics and alerts | TBD | Not configured | TBD | TBD |
| Health checks and rollback | TBD | Not configured | TBD | TBD |

### Operational runbook to complete after rollout

Document provider dashboard links, rate-limit thresholds, WAF exception process, temporary IP-block expiry policy, alert recipients, secret rotation steps, rollback procedure, and incident escalation contact. Store account-specific values in the organization's approved operations system, not in this public repository.
