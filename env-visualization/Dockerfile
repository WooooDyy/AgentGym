FROM node:20-alpine

WORKDIR /app

# Copy package files
COPY package.json package-lock.json* ./

# Install dependencies
RUN npm install

# Copy source files
COPY . .

# Expose port
EXPOSE 5173

# Run in development mode with host binding
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]
