"use client";

import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Separator } from "@/components/ui/separator";
import { ArrowLeft, Leaf } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { API_BASE } from "../config.ts";


const IrrigationPage: React.FC = () => {
  const navigate = useNavigate();
  const [location, setLocation] = useState<string>("");
  const [searchResults, setSearchResults] = useState<any[]>([]);
  const [moisture, setMoisture] = useState<number>(50);
  const [temperature, setTemperature] = useState<number | null>(null);
  const [pressure, setPressure] = useState<number | null>(null);
  const [altitude, setAltitude] = useState<number | null>(null);
  const [rainExpected, setRainExpected] = useState<boolean | null>(null);
  const [advice, setAdvice] = useState<string>("");

  const handleSearch = async (query: string) => {
    setLocation(query);
    if (query.length > 2) {
      const response = await fetch(
        `https://nominatim.openstreetmap.org/search?format=json&q=${query}`
      );
      const data = await response.json();
      setSearchResults(data);
    } else {
      setSearchResults([]);
    }
  };

  const selectLocation = (place: any) => {
    setLocation(place.display_name);
    setSearchResults([]);
  };

  const fetchWeatherData = async (): Promise<void> => {
    try {
      const response = await fetch(`${API_BASE}/check_weather?location=${location}`);
      const data: { rain_expected: boolean; temperature: number; pressure: number; altitude: number } = await response.json();

      setRainExpected(data.rain_expected);
      setTemperature(data.temperature);
      setPressure(data.pressure);
      setAltitude(data.altitude);

      if (data.rain_expected) {
        setAdvice("No irrigation needed due to expected rain.");
      }
    } catch (error) {
      console.error("Error fetching weather data:", error);
      setAdvice("Failed to fetch weather data.");
    }
  };

  const getIrrigationAdvice = async (): Promise<void> => {
    if (rainExpected) return;
    setAdvice("Fetching irrigation advice...");

    try {
      const response = await fetch(`${API_BASE}/predict/irrigation`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          temperature,
          soil_moisture: moisture,
          pressure,
          altitude,
        }),
      });

      if (!response.ok) {
        throw new Error(`Server responded with ${response.status}`);
      }

      const data: { prediction: string } = await response.json();
      setAdvice(data.prediction);
    } catch (error) {
      console.error("Error fetching irrigation advice:", error);
      setAdvice("Failed to get advice. Please try again.");
    }
  };

  return (
    <div className="min-h-screen flex flex-col bg-gray-50">
      <header className="px-4 py-2 border-b flex items-center justify-between bg-white shadow-sm">
        <div className="flex items-center gap-1 font-medium">
          <Leaf className="h-5 w-5 text-green-600" />
          <span className="text-md">FarmFriend</span>
        </div>
        <button
          onClick={() => navigate("/")}
          className="flex items-center text-green-600 text-sm hover:text-green-700 transition"
        >
          <ArrowLeft className="h-4 w-4 mr-1" />
          Home
        </button>
      </header>

      <main className="flex flex-1 items-center justify-end p-6 bg-[url('/latest.png')] bg-cover bg-center bg-no-repeat relative before:absolute before:inset-0 before:bg-black/40">
        <Card className="w-full max-w-md shadow-xl p-6 bg-blue-100 rounded-lg relative z-10 mr-12">
          <CardHeader>
            <CardTitle className="text-center text-xl font-semibold">Irrigation Advisor</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {!temperature && (
                <div>
                  <Label>Enter Your Location</Label>
                  <Input
                    type="text"
                    value={location}
                    onChange={(e) => handleSearch(e.target.value)}
                    placeholder="Search location..."
                  />
                  {searchResults.length > 0 && (
                    <ul className="border rounded-md bg-white mt-1 shadow-md">
                      {searchResults.map((place, index) => (
                        <li
                          key={index}
                          className="px-3 py-2 hover:bg-gray-100 cursor-pointer"
                          onClick={() => selectLocation(place)}
                        >
                          {place.display_name}
                        </li>
                      ))}
                    </ul>
                  )}
                  <Button onClick={fetchWeatherData} className="w-full mt-2 bg-blue-600 hover:bg-blue-700">
                    Fetch Weather Data
                  </Button>
                </div>
              )}
              {temperature !== null && (
                <>
                  <p className="text-center font-semibold">Temperature: {temperature}°C</p>
                  <p className="text-center font-semibold">Pressure: {pressure} hPa</p>
                  <p className="text-center font-semibold">Altitude: {altitude} m</p>
                  <Separator />
                  {!rainExpected && (
                    <div>
                      <Label>Soil Moisture (%)</Label>
                      <Input
                        type="text"
                        value={moisture}
                        onChange={(e) => {
                          let value = e.target.value.replace(/\D/g, ""); // Remove non-numeric characters
                          let num = value ? Number(value) : 0;

                          if (num < 0 || num > 100) {
                            alert("Please enter a value between 0 and 100.");
                            return;
                          }

                          setMoisture(num);
                        }}
                        placeholder="Enter soil moisture (0-100%)"
                        className="appearance-none border rounded-md px-3 py-2 w-full focus:ring-2 focus:ring-green-500"
                      />
                      <Button onClick={getIrrigationAdvice} className="w-full mt-2 bg-green-600 hover:bg-green-700">
                        Get Advice
                      </Button>
                    </div>
                  )}
                  {advice && <p className="text-center mt-4 font-semibold text-gray-800">{advice}</p>}
                </>
              )}
            </div>
          </CardContent>
        </Card>
      </main>
    </div>
  );
};

export default IrrigationPage;
