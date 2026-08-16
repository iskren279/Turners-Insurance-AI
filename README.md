# Car Insurance API

REST API for calculating car insurance-related values and ratings.

## Installation

Clone the repository and install the project dependencies:

```bash
npm install
```

## Running the Application

Create a `.env` file in the project root:

```env
PORT=3000
```

Start the server:

```bash
npm start
```

The API will run on:

```text
http://localhost:3000
```

## Running Tests

Run the test suite with:

```bash
npm test
```

## Project Structure

```text
src/
├── app.js
├── server.js
├── routes/
└── services/

tests/
```

### `src/app.js`

Configures the Express application and middleware.

### `src/server.js`

Loads environment variables and starts the HTTP server.

### `src/routes/`

Contains the API routes.

### `src/services/`

Contains the business logic for each API.

### `tests/`

Contains unit and API tests.

## APIs

The project contains four APIs:

1. **Car Value**
2. **Risk Rating**
3. **Quote**
4. **Discount Rate**

Details for each API will be added as they are implemented.

## Git Workflow

Do not work directly on `main`.

Create a feature branch for your work:

```bash
git checkout main
git pull origin main
git checkout -b feature/<your-api-name>
```

After completing your work:

```bash
git add .
git commit -m "feat: <description>"
git push -u origin feature/<your-api-name>
```

Create a Pull Request on GitHub for review before merging into `main`.

## Environment Variables

The `.env` file is not committed to GitHub.

Use `.env.example` as a template for the required environment variables.

You can create your local `.env` file with:

```bash
cp .env.example .env
```

## Contributors

- **Iskren Petrov** — GitHub: [@iskren279]
- **Fateha Momen** — GitHub: [@fatehamomen85-lang]
- **Banphot Uthahan** — GitHub: [@bu-missionreadyhq]
