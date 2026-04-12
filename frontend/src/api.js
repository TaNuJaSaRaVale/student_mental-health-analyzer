const API_URL = import.meta.env.PROD ? "" : "http://localhost:8000";

export const getPrediction = async (formData) => {
  try {
    const response = await fetch(`${API_URL}/predict`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(formData),
    });

    if (!response.ok) {
      throw new Error("Network response was not ok");
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error("Error fetching prediction:", error);
    throw error;
  }
};
