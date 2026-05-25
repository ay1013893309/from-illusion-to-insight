"""
Expert debate orchestration for defect prediction.
"""
import logging
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
import time

from .wrapper import OpenAIWrapper
from ..prompts.loader import load_all_prompts
from ..analysis.verdict_parser import parse_confidence, parse_judge_verdict

logger = logging.getLogger(__name__)


@dataclass
class DebateRound:
    """Represents a single debate round with all expert responses."""
    round_number: int
    analyzer_response: Optional[str] = None
    proposer_response: Optional[str] = None
    skeptic_response: Optional[str] = None
    judge_response: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


class ExpertDebateSystem:
    """Orchestrates debate between multiple LLM experts for defect prediction."""
    
    def __init__(
        self,
        llm_client: OpenAIWrapper,
        analyzer_model: str = "deepseek-v4-flash",
        proposer_model: str = "deepseek-v4-flash",
        skeptic_model: str = "deepseek-v4-flash",
        judge_model: str = "deepseek-v4-flash",
        temperature: float = 0.7,
        max_rounds: int = 3,
    ):
        """
        Initialize the expert debate system.
        
        Args:
            llm_client: OpenAIWrapper instance for API calls
            analyzer_model: Model name for Analyzer expert
            proposer_model: Model name for Proposer expert
            skeptic_model: Model name for Skeptic expert
            judge_model: Model name for Judge expert
            temperature: Sampling temperature for all experts
            max_rounds: Maximum debate rounds (Proposer/Skeptic exchanges)
        """
        self.llm_client = llm_client
        self.analyzer_model = analyzer_model
        self.proposer_model = proposer_model
        self.skeptic_model = skeptic_model
        self.judge_model = judge_model
        self.temperature = temperature
        self.max_rounds = max_rounds
        self.prompt_loader = load_all_prompts()
        self.debate_history: List[DebateRound] = []
    
    def run_analyzer(
        self,
        diff_text: str,
        src1_context: str,
        src2_context: str,
        previous_status: str,
    ) -> str:
        """
        Run the Analyzer expert to provide detailed evidence.
        
        Args:
            diff_text: The unified diff
            src1_context: Old version context
            src2_context: New version context
            previous_status: Previous label (BENIGN or DEFECTIVE)
        
        Returns:
            Analyzer's response
        """
        prompt = self.prompt_loader.get_analyzer(diff_text, src1_context, src2_context, previous_status)
        logger.info("Running Analyzer expert...")
        
        response = self.llm_client.create_completion(
            model=self.analyzer_model,
            messages=[
                {"role": "system", "content": prompt["system"]},
                {"role": "user", "content": prompt["user"]},
            ],
            temperature=self.temperature,
        )
        
        logger.debug(f"Analyzer response:\n{response}")
        return response
    
    def run_proposer(
        self,
        diff_text: str,
        context: str,
        analyzer_response: str,
        previous_status: str,
        skeptic_history: Optional[str] = None,
    ) -> str:
        """
        Run the Proposer expert to make initial defect judgment.
        
        Args:
            diff_text: The unified diff
            context: Code context
            analyzer_response: Response from Analyzer
            previous_status: Previous label
            skeptic_history: Previous Skeptic responses (for debate rounds)
        
        Returns:
            Proposer's response
        """
        prompt = self.prompt_loader.get_proposer(diff_text, context, analyzer_response, previous_status)
        user_msg = prompt["user"]
        
        if skeptic_history:
            user_msg += f"\n\nSceptic's previous critique:\n{skeptic_history}"
        
        logger.info("Running Proposer expert...")
        
        response = self.llm_client.create_completion(
            model=self.proposer_model,
            messages=[
                {"role": "system", "content": prompt["system"]},
                {"role": "user", "content": user_msg},
            ],
            temperature=self.temperature,
        )
        
        logger.debug(f"Proposer response:\n{response}")
        return response
    
    def run_skeptic(
        self,
        diff_text: str,
        context: str,
        proposer_response: str,
    ) -> str:
        """
        Run the Skeptic expert to provide adversarial critique.
        
        Args:
            diff_text: The unified diff
            context: Code context
            proposer_response: Response from Proposer
        
        Returns:
            Skeptic's response
        """
        prompt = self.prompt_loader.get_skeptic(diff_text, context, proposer_response)
        logger.info("Running Skeptic expert...")
        
        response = self.llm_client.create_completion(
            model=self.skeptic_model,
            messages=[
                {"role": "system", "content": prompt["system"]},
                {"role": "user", "content": prompt["user"]},
            ],
            temperature=self.temperature,
        )
        
        logger.debug(f"Skeptic response:\n{response}")
        return response
    
    def run_judge(
        self,
        diff_text: str,
        context: str,
        proposer_response: str,
        skeptic_response: str,
        previous_status: str,
    ) -> Tuple[str, int]:
        """
        Run the Judge expert to make final defect prediction.
        
        Args:
            diff_text: The unified diff
            context: Code context
            proposer_response: Response from Proposer
            skeptic_response: Response from Skeptic
            previous_status: Previous label
        
        Returns:
            Tuple of (judge_response, confidence_percentage)
        """
        prompt = self.prompt_loader.get_judge(
            diff_text, context, proposer_response, skeptic_response, previous_status
        )
        logger.info("Running Judge expert...")
        
        response = self.llm_client.create_completion(
            model=self.judge_model,
            messages=[
                {"role": "system", "content": prompt["system"]},
                {"role": "user", "content": prompt["user"]},
            ],
            temperature=self.temperature,
        )
        
        logger.debug(f"Judge response:\n{response}")
        
        # Parse confidence from response
        confidence = self._extract_confidence(response)
        return response, confidence
    
    def run_debate(
        self,
        diff_text: str,
        src1_context: str,
        src2_context: str,
        context: str,
        previous_status: str,
    ) -> Tuple[str, int, List[DebateRound]]:
        """
        Run the full debate process: Analyzer -> Proposer + Skeptic debate -> Judge.
        
        Args:
            diff_text: The unified diff
            src1_context: Old version context
            src2_context: New version context
            context: General context
            previous_status: Previous label
        
        Returns:
            Tuple of (final_prediction, confidence, debate_history)
        """
        logger.info("Starting expert debate...")
        
        # Round 0: Analyzer runs once
        analyzer_response = self.run_analyzer(diff_text, src1_context, src2_context, previous_status)
        round_0 = DebateRound(round_number=0, analyzer_response=analyzer_response)
        self.debate_history.append(round_0)
        
        proposer_response = ""
        skeptic_response = ""
        
        # Rounds 1 to max_rounds: Proposer and Skeptic debate
        for round_num in range(1, self.max_rounds + 1):
            logger.info(f"Debate round {round_num}/{self.max_rounds}")
            
            proposer_response = self.run_proposer(
                diff_text, context, analyzer_response, previous_status, skeptic_response
            )
            skeptic_response = self.run_skeptic(diff_text, context, proposer_response)
            
            round_obj = DebateRound(
                round_number=round_num,
                proposer_response=proposer_response,
                skeptic_response=skeptic_response,
            )
            self.debate_history.append(round_obj)
        
        # Final: Judge makes decision
        judge_response, confidence = self.run_judge(
            diff_text, context, proposer_response, skeptic_response, previous_status
        )
        
        final_round = DebateRound(round_number=self.max_rounds + 1, judge_response=judge_response)
        self.debate_history.append(final_round)
        
        # Extract final prediction from judge response
        final_prediction = self._extract_prediction(judge_response)
        
        logger.info(f"Debate complete. Final prediction: {final_prediction} (confidence: {confidence}%)")
        
        return final_prediction, confidence, self.debate_history
    
    @staticmethod
    def _extract_confidence(response: str) -> int:
        """Extract confidence percentage from Judge response."""
        confidence = parse_confidence(response)
        return confidence if confidence is not None else 50  # Default to 50% if not found
    
    @staticmethod
    def _extract_prediction(response: str) -> str:
        """Extract final prediction (BENIGN or DEFECTIVE) from Judge response."""
        verdict_str, _ = parse_judge_verdict(response)
        if verdict_str is not None:
            return verdict_str
        return "INCONCLUSIVE"
