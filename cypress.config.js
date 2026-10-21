const { defineConfig } = require("cypress");

module.exports = defineConfig({
  allowCypressEnv: false,

  e2e: {
    baseUrl: "http://localhost:8501",
    supportFile: "cypress/support/e2e.js",
    specPattern: "cypress/e2e/**/*.cy.js",

    setupNodeEvents(on, config) {
      return config;
    },
  },

  defaultCommandTimeout: 60000,
  pageLoadTimeout: 120000,
  requestTimeout: 120000,
  responseTimeout: 120000,

  viewportWidth: 1440,
  viewportHeight: 900,
  video: true,
  screenshotOnRunFailure: true,
});