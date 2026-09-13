# The portability contract for a Linux host. Version 1 of the platform runs the
# app as a native process, this image is what CI builds on publish.
# No model and no Postgres inside: both are injected as URLs.

FROM oven/bun:1.3-alpine AS builder
WORKDIR /app
ENV NEXT_TELEMETRY_DISABLED=1
COPY . .
RUN bun run aias:build

FROM node:24-alpine AS runner
WORKDIR /app
ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1
ENV HOSTNAME=0.0.0.0
ENV PORT=3000
RUN addgroup -S app && adduser -S app -G app
COPY --from=builder --chown=app:app /app/.next/standalone ./
USER app
EXPOSE $PORT
CMD ["node", "server.js"]
