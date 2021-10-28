from tqdm import tqdm
from farm.train import Trainer
from farm.eval import Evaluator
from farm.visual.ascii.images import GROWING_TREE
import logging
import sys
import optuna

logger = logging.getLogger(__name__)

class TrainerOptuna(Trainer):
    def train(self, trial):
        """
        Perform the training procedure and allow optuna to stop the training if
        accuracy of classification of downstream NLP task does not improve after
        a number of iterations
        """

        # connect the prediction heads with the right output from processor
        self.model.connect_heads_with_processor(self.data_silo.processor.tasks, require_labels=True)

        # Check that the tokenizer fits the language model
        self.model.verify_vocab_size(vocab_size=len(self.data_silo.processor.tokenizer))

        logger.info(f"\n {GROWING_TREE}")
        self.model.train()

        do_stopping = False
        evalnr = 0
        loss = 0

        resume_from_step = self.from_step

        for epoch in range(self.from_epoch, self.epochs):
            self.from_epoch = epoch
            train_data_loader = self.data_silo.get_data_loader("train")
            progress_bar = tqdm(train_data_loader)
            for step, batch in enumerate(progress_bar):
                # when resuming training from a checkpoint, we want to fast forward to the step of the checkpoint
                if resume_from_step and step <= resume_from_step:
                    if resume_from_step == step:
                        resume_from_step = None
                    continue

                progress_bar.set_description(f"Train epoch {epoch}/{self.epochs} (Cur. train loss: {loss:.4f})")

                # Move batch of samples to device
                batch = {key: batch[key].to(self.device) for key in batch}

                # Forward pass through model
                logits = self.model.forward(**batch)
                per_sample_loss = self.model.logits_to_loss(logits=logits, global_step=self.global_step, **batch)

                loss = self.backward_propagate(per_sample_loss, step)

                # Perform  evaluation
                if self.evaluate_every != 0 and self.global_step % self.evaluate_every == 0 and self.global_step != 0:
                    # When using StreamingDataSilo, each evaluation creates a new instance of
                    # dev_data_loader. In cases like training from scratch, this could cause
                    # some variance across evaluators due to the randomness in word masking.
                    dev_data_loader = self.data_silo.get_data_loader("dev")
                    if dev_data_loader is not None:
                        evaluator_dev = Evaluator(
                            data_loader=dev_data_loader, tasks=self.data_silo.processor.tasks, device=self.device
                        )
                        evalnr += 1
                        result = evaluator_dev.eval(self.model)
                        evaluator_dev.log_results(result, "Dev", self.global_step)

                        intermediate_value = result[0]["acc"]
                        trial.report(intermediate_value, step)

                        if trial.should_prune():
                            raise optuna.TrialPruned()

                        # save the current state as a checkpoint before exiting if a SIGTERM signal is received
                if self.sigterm_handler and self.sigterm_handler.kill_now:
                    logger.info("Received a SIGTERM signal. Saving the current train state as a checkpoint ...")
                    self._save()
                    sys.exit(0)

                # save a checkpoint and continue train
                if self.checkpoint_every and step % self.checkpoint_every == 0:
                    self._save()

            if do_stopping:
                break

            self.global_step += 1
            self.from_step = step + 1

        # Eval on test set
        test_data_loader = self.data_silo.get_data_loader("test")
        if test_data_loader is not None:
            evaluator_test = Evaluator(
                data_loader=test_data_loader, tasks=self.data_silo.processor.tasks, device=self.device
            )
            result = evaluator_test.eval(self.model)
            evaluator_test.log_results(result, "Test", self.global_step)
        return self.model


