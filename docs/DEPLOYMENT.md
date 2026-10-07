# 🚀 Free Deployment Guide for PassShield

This guide explains how to deploy **PassShield** to the cloud for **100% FREE** with HTTPS.

---

## 🌟 Option 1: Render (Recommended — Free Web Service)

Render allows you to host Python Flask apps with ML models for free.

### Step 1: Push Code to GitHub
1. Open terminal in `password_analyzer`:
   ```bash
   git init
   git add .
   git commit -m "PassShield release v1.0"
   ```
2. Create a new repository on GitHub (e.g., `passshield-password-analyzer`).
3. Push your repository to GitHub:
   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/passshield-password-analyzer.git
   git branch -M main
   git push -u origin main
   ```

### Step 2: Deploy on Render
1. Go to **[https://render.com](https://render.com)** and sign up for a free account.
2. Click **"New +"** $\to$ **"Web Service"**.
3. Connect your GitHub repository (`passshield-password-analyzer`).
4. Fill in the deployment details:
   - **Name:** `passshield`
   - **Environment:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app`
   - **Instance Type:** `Free`
5. Click **"Advanced"** $\to$ **"Add Environment Variable"**:
   - **Key:** `GROQ_API_KEY`
   - **Value:** `gsk_your_groq_api_key_here`
6. Click **"Create Web Service"**.

🎉 Your live app will be published at `https://passshield.onrender.com` in 2-3 minutes!

---

## 🚀 Option 2: Hugging Face Spaces (Free Docker/Gradio/Flask)

1. Go to **[https://huggingface.co/spaces](https://huggingface.co/spaces)**.
2. Click **"Create new Space"**.
3. Select **Docker** or **Gradio/Flask**.
4. Set Space SDK to `Docker` (or Blank).
5. Push your code. It will build and serve for free at `https://huggingface.co/spaces/YOUR_USERNAME/passshield`.

---

## ⚡ Option 3: Railway.app (Free Trial)

1. Sign up at **[https://railway.app](https://railway.app)**.
2. Click **"New Project"** $\to$ **"Deploy from GitHub repo"**.
3. Add `GROQ_API_KEY` in Environment Variables.
4. Railway detects `Procfile` and deploys automatically!

---

## Self-hosted Nginx security configuration

The project includes an Nginx reverse-proxy configuration for deployments where you manage the host. It provides HTTPS redirect/TLS settings, response security headers, a 16 KiB request-body limit, connection limits, no-store responses, and per-IP API throttling. It is not automatically active on Render, Hugging Face Spaces, or Railway; those hosted options require their own edge/WAF settings.

### Render deployment note

For a service deployed at `*.onrender.com`, do not install the Nginx site file: Render terminates TLS, redirects HTTP to HTTPS, and supplies platform-level DDoS protection automatically ([TLS](https://render.com/docs/tls), [DDoS protection](https://render.com/docs/ddos-protection)). The Nginx rate limits and response headers do not apply to your Render service. Render's edge DDoS protection does not configure per-route application rate limits. This app currently has no such rate limiter, so `/analyze`, `/improve`, and `/export_report` are not protected by the Nginx limits on Render.

To add configurable edge rules while staying on Render, use a custom domain through a customer-controlled CDN/WAF (for example, Cloudflare) and configure its rate-limit/WAF rules there. Ensure requests cannot bypass those rules through the `onrender.com` hostname: Render supports disabling the default Render subdomain in the service's Custom Domains settings ([custom domains](https://render.com/docs/custom-domains)). Configure this only after the custom domain and proxy are working. Otherwise, application-level rate limiting requires adding middleware/code to this Flask app. Render also describes targeted HTTP/API abuse as requiring application-level defenses ([DDoS guidance](https://render.com/articles/how-render-handles-ddos-attacks)).

1. Run Gunicorn or another production WSGI server on a private listener at `127.0.0.1:8000`. Do not expose Flask's development server.
2. Install `deploy/nginx/http-security.conf` in a location included inside Nginx's `http {}` block (commonly `/etc/nginx/conf.d/`).
3. Install `deploy/nginx/passshield-site.conf` as a site configuration. Replace `passshield.example.com` and the certificate paths with the real hostname and certificate. Obtain a valid certificate before enabling the HTTPS server block.
4. Run `nginx -t` and reload Nginx only if the configuration is valid.
5. Verify HTTPS, the response headers, and `429` responses after exceeding the configured per-IP limits. Review logs to ensure request bodies are not captured.

If a CDN is placed in front of Nginx, configure real-client-IP handling using only that CDN's trusted IP ranges before enabling the rate limits. The supplied limits are 120 requests/minute (burst 10) for `/analyze` and `/export_report`, and 5 requests/minute (burst 2) for `/improve`. Tune them against real usage. Nginx host configuration does not replace the CDN's WAF or DDoS protections.
