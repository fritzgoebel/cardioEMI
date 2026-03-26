// config-manager.js - Handle YAML configuration updates

class ConfigManager {
    constructor(apiBase = '/api') {
        this.apiBase = apiBase;
        this.configFile = 'input_pepe36_colored.yml';
    }

    setConfigFile(configFile) {
        this.configFile = configFile;
    }

    async updateConfig(updates) {
        const response = await fetch(`${this.apiBase}/config`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                file: this.configFile,
                updates: updates
            })
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Failed to update config');
        }

        return await response.json();
    }

    async getConfig() {
        const response = await fetch(`${this.apiBase}/config?file=${this.configFile}`);

        if (!response.ok) {
            throw new Error('Failed to load config');
        }

        return await response.json();
    }

    async updatePetscBddcConfig(bddcConfig) {
        // Update the nested petsc_bddc configuration in the YAML file
        const response = await fetch(`${this.apiBase}/config/petsc_bddc`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                file: this.configFile,
                petsc_bddc: bddcConfig
            })
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Failed to update PETSc BDDC config');
        }

        return await response.json();
    }

    async updateGinkgoConfig(ginkgoConfig) {
        // Update the nested ginkgo configuration in the YAML file
        const response = await fetch(`${this.apiBase}/config/ginkgo`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                file: this.configFile,
                ginkgo: ginkgoConfig
            })
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Failed to update Ginkgo config');
        }

        return await response.json();
    }
}
