import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import axios from "axios";
import LandingPage from "../../src/LandingPage";

jest.mock("axios");

describe("LandingPage", () => {
  test("renders title", () => {
    render(<LandingPage onStart={() => {}} />);
    expect(screen.getByText("DLSU Handbook Assistant")).toBeInTheDocument();
  });

  test("renders subtitle", () => {
    render(<LandingPage onStart={() => {}} />);
    expect(screen.getByText(/Your AI guide/i)).toBeInTheDocument();
  });

  test("renders start button", () => {
    render(<LandingPage onStart={() => {}} />);
    expect(screen.getByText(/Start chatting/i)).toBeInTheDocument();
  });

  test("renders warning box", () => {
    render(<LandingPage onStart={() => {}} />);
    expect(screen.getByText(/unofficial/i)).toBeInTheDocument();
  });

  test("shows stats when loaded", async () => {
    axios.get.mockResolvedValueOnce({
      data: { total_visits: 123, total_questions: 456 }
    });
    render(<LandingPage onStart={() => {}} />);
    await waitFor(() => {
      expect(screen.getByText("123")).toBeInTheDocument();
      expect(screen.getByText("456")).toBeInTheDocument();
    });
  });

  test("shows dash when stats fail", async () => {
    axios.get.mockRejectedValueOnce(new Error("Network error"));
    render(<LandingPage onStart={() => {}} />);
    await waitFor(() => {
      const dashes = screen.getAllByText("—");
      expect(dashes.length).toBeGreaterThan(0);
    });
  });

  test("clicking start opens ToS modal", () => {
    render(<LandingPage onStart={() => {}} />);
    fireEvent.click(screen.getByText(/Start chatting/i));
    expect(screen.getByText("Terms of Service")).toBeInTheDocument();
  });

  test("agreeing to ToS calls onStart", () => {
    const onStart = jest.fn();
    render(<LandingPage onStart={onStart} />);
    fireEvent.click(screen.getByText(/Start chatting/i));
    fireEvent.click(screen.getByText(/I Agree/i));
    expect(onStart).toHaveBeenCalledTimes(1);
  });

  test("declining ToS closes modal", () => {
    render(<LandingPage onStart={() => {}} />);
    fireEvent.click(screen.getByText(/Start chatting/i));
    expect(screen.getByText("Terms of Service")).toBeInTheDocument();
    fireEvent.click(screen.getByText("Decline"));
    expect(screen.queryByText("Terms of Service")).not.toBeInTheDocument();
  });

  test("ToS contains required sections", () => {
    render(<LandingPage onStart={() => {}} />);
    fireEvent.click(screen.getByText(/Start chatting/i));
    expect(screen.getByText(/Acceptance/i)).toBeInTheDocument();
    expect(screen.getByText(/Accuracy Disclaimer/i)).toBeInTheDocument();
    expect(screen.getByText(/Limitation of Liability/i)).toBeInTheDocument();
  });
});
