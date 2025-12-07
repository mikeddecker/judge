# Stage 1: Build the Vue app
FROM node:20-alpine AS builder

WORKDIR /app

# Install dependencies
COPY package*.json ./
RUN npm install

# Copy source code
COPY . .

# Build production-ready static files
RUN npm run build

# Stage 2: Serve with nginx
FROM nginx:alpine

# Copy built files from builder
COPY --from=builder /app/dist /usr/share/nginx/html

# Optional: copy custom nginx config
# COPY nginx.conf /etc/nginx/nginx.conf

# Expose port (container internal)
EXPOSE ${WEB_DOCKER_PORT:-80}

# Start nginx in foreground
CMD ["nginx", "-g", "daemon off;"]

