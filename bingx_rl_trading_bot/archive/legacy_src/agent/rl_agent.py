"""강화학습 에이전트 (PPO)"""

from pathlib import Path
from typing import Dict, Any, Optional

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from loguru import logger


class RLAgent:
    """
    PPO 강화학습 에이전트
    """

    def __init__(
        self,
        env,
        config: Dict[str, Any] = None,
        model_dir: str = None
    ):
        """
        Args:
            env: Gymnasium 환경
            config: 에이전트 설정
            model_dir: 모델 저장 디렉토리
        """
        self.env = env
        self.config = config or self._default_config()

        if model_dir is None:
            project_root = Path(__file__).parent.parent.parent
            model_dir = project_root / 'data' / 'trained_models'

        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.model = None
        self.training_stats = {
            'episodes': 0,
            'timesteps': 0,
            'best_reward': float('-inf')
        }

    @staticmethod
    def _default_config() -> Dict[str, Any]:
        """기본 설정"""
        return {
            'learning_rate': 0.0003,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'n_steps': 2048,
            'batch_size': 64,
            'n_epochs': 10,
            'ent_coef': 0.01,
            'vf_coef': 0.5,
            'max_grad_norm': 0.5,
            'verbose': 1
        }

    def create_model(self, policy: str = "MlpPolicy") -> None:
        """
        PPO 모델 생성

        Args:
            policy: 정책 네트워크 타입
        """
        logger.info("Creating PPO model...")

        # Monitor로 환경 래핑 (성능 추적)
        from stable_baselines3.common.monitor import Monitor
        monitored_env = Monitor(self.env)

        # 환경을 벡터화된 환경으로 래핑
        vec_env = DummyVecEnv([lambda: monitored_env])

        self.model = PPO(
            policy=policy,
            env=vec_env,
            learning_rate=self.config.get('learning_rate', 0.0003),
            gamma=self.config.get('gamma', 0.99),
            gae_lambda=self.config.get('gae_lambda', 0.95),
            clip_range=self.config.get('clip_range', 0.2),
            n_steps=self.config.get('n_steps', 2048),
            batch_size=self.config.get('batch_size', 64),
            n_epochs=self.config.get('n_epochs', 10),
            ent_coef=self.config.get('ent_coef', 0.01),
            vf_coef=self.config.get('vf_coef', 0.5),
            max_grad_norm=self.config.get('max_grad_norm', 0.5),
            verbose=self.config.get('verbose', 1),
            device='cpu',  # MLP 정책은 CPU가 더 효율적
            tensorboard_log=str(self.model_dir / 'tensorboard')
        )

        logger.info("PPO model created")

    def train(
        self,
        total_timesteps: int = 100000,
        eval_env=None,
        eval_freq: int = 10000,
        save_freq: int = 50000,
        callback=None
    ) -> None:
        """
        모델 훈련

        Args:
            total_timesteps: 총 훈련 타임스텝
            eval_env: 평가 환경
            eval_freq: 평가 빈도
            save_freq: 저장 빈도
            callback: 커스텀 콜백
        """
        if self.model is None:
            self.create_model()

        logger.info(f"Starting training for {total_timesteps} timesteps...")

        # 콜백 설정
        callbacks = []

        if eval_env is not None:
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=str(self.model_dir / 'best_model'),
                log_path=str(self.model_dir / 'eval_logs'),
                eval_freq=eval_freq,
                deterministic=True,
                render=False
            )
            callbacks.append(eval_callback)

        if callback is not None:
            callbacks.append(callback)

        # 훈련 진행
        try:
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=callbacks if callbacks else None,
                log_interval=100,
                progress_bar=True
            )

            # 최종 모델 저장
            self.save_model('final_model')

            logger.info("Training completed successfully")

        except KeyboardInterrupt:
            logger.warning("Training interrupted by user")
            self.save_model('interrupted_model')
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise

    def predict(
        self,
        observation,
        deterministic: bool = True
    ):
        """
        행동 예측

        Args:
            observation: 관측 값
            deterministic: 결정적 행동 여부

        Returns:
            (행동, 상태)
        """
        if self.model is None:
            raise ValueError("Model not created or loaded")

        action, state = self.model.predict(observation, deterministic=deterministic)
        return action, state

    def save_model(self, filename: str) -> None:
        """
        모델 저장

        Args:
            filename: 파일명 (확장자 제외)
        """
        if self.model is None:
            logger.warning("No model to save")
            return

        filepath = self.model_dir / f"{filename}.zip"
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filename: str) -> None:
        """
        모델 로드

        Args:
            filename: 파일명 (확장자 제외)
        """
        # best_model은 디렉토리 안에 .zip이 있음
        filepath = self.model_dir / f"{filename}.zip"
        if not filepath.exists():
            # 디렉토리 내부 확인
            filepath = self.model_dir / filename / f"{filename}.zip"

        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")

        # 환경 래핑
        vec_env = DummyVecEnv([lambda: self.env])

        self.model = PPO.load(filepath, env=vec_env)
        logger.info(f"Model loaded from {filepath}")

    def evaluate(
        self,
        n_episodes: int = 10,
        deterministic: bool = True
    ) -> Dict[str, float]:
        """
        모델 평가

        Args:
            n_episodes: 평가 에피소드 수
            deterministic: 결정적 행동 여부

        Returns:
            평가 통계
        """
        if self.model is None:
            raise ValueError("Model not created or loaded")

        logger.info(f"Evaluating model for {n_episodes} episodes...")

        episode_rewards = []
        episode_lengths = []

        for episode in range(n_episodes):
            obs, info = self.env.reset()
            done = False
            episode_reward = 0
            episode_length = 0

            while not done:
                action, _ = self.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                episode_reward += reward
                episode_length += 1

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

            logger.debug(f"Episode {episode + 1}: Reward={episode_reward:.2f}, Length={episode_length}")

        # 통계 계산
        import numpy as np
        stats = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards),
            'mean_length': np.mean(episode_lengths)
        }

        logger.info(f"Evaluation results: Mean reward={stats['mean_reward']:.2f} "
                   f"(±{stats['std_reward']:.2f})")

        return stats


class TradingCallback(BaseCallback):
    """
    훈련 중 거래 통계 로깅 콜백
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        """각 스텝마다 호출"""
        # 에피소드 종료 시 통계 기록
        for info in self.locals.get('infos', []):
            if 'episode' in info:
                self.episode_rewards.append(info['episode']['r'])
                self.episode_lengths.append(info['episode']['l'])

                # 로깅
                if self.verbose > 0:
                    logger.info(f"Episode finished - Reward: {info['episode']['r']:.2f}, "
                               f"Length: {info['episode']['l']}")

        return True

    def _on_rollout_end(self) -> None:
        """롤아웃 종료 시 호출"""
        if len(self.episode_rewards) > 0:
            import numpy as np
            logger.info(f"Last 10 episodes - Mean reward: {np.mean(self.episode_rewards[-10:]):.2f}")
