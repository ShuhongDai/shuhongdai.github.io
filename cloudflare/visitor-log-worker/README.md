# Private Visitor Log for GitHub Pages

This Worker records only a minimal visit record:
- timestamp
- city
- country
- path
- site name

## 1. Create the D1 table

Run this SQL in Cloudflare D1:

```sql
.read schema.sql
```

## 2. Configure the Worker

Use `wrangler.toml.example` as your template and set:
- `database_id`
- `ADMIN_TOKEN`

## 3. Deploy the Worker

Expected endpoints:
- `POST /collect`
- `GET /admin?token=YOUR_TOKEN`
- `GET /admin?token=YOUR_TOKEN&format=json`

## 4. Connect the GitHub Pages site

In `_config.yml` set:

```yml
visitor_logging:
  enabled: true
  endpoint: https://your-worker.workers.dev/collect
  site_name: shuhongdai-homepage
```

## 5. View the logs privately

Open:

```text
https://your-worker.workers.dev/admin?token=YOUR_TOKEN
```

This page is not linked publicly from the website.
